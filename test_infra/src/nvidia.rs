// Copyright (c) 2026 Crusoe Energy Systems LLC
//
// SPDX-License-Identifier: Apache-2.0
//

//! Checks of an NVIDIA GPU passed through to a guest: nvidia-smi views and
//! properties.

use crate::{Guest, SshCommandError};

/// Lines that report a field nvidia-smi could not read: it prints "Unknown
/// Error" ("[Unknown Error]" in CSV output) and still exits 0.
pub fn nvidia_smi_error_lines(output: &str) -> Vec<&str> {
    output
        .lines()
        .filter(|line| line.contains("Unknown Error"))
        .collect()
}

/// nvidia-smi views every Grace GPU (Hopper, Blackwell) supports, the C2C
/// link included: each must exit 0 and report no "Unknown Error".
pub const NVIDIA_SMI_VIEWS: &[&str] = &[
    "nvidia-smi --version",
    "nvidia-smi",
    "nvidia-smi -L",
    "nvidia-smi -q",
    "nvidia-smi -q -x",
    "nvidia-smi -q -d SUPPORTED_CLOCKS",
    "nvidia-smi topo -m",
    "nvidia-smi topo -p2p r",
    "nvidia-smi topo -p2p w",
    "nvidia-smi topo -p2p n",
    "nvidia-smi topo -p2p a",
    "nvidia-smi topo -p2p p",
    "nvidia-smi dmon -c 3 -s pucvmet",
    "nvidia-smi pmon -c 1",
    "nvidia-smi --query-retired-pages=gpu_uuid,retired_pages.address,retired_pages.cause --format=csv",
    "nvidia-smi --query-remapped-rows=gpu_uuid,remapped_rows.correctable,\
     remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure --format=csv",
    "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
    "nvidia-smi c2c -s",
];

/// Views of hardware not every GPU has (NVLink, MIG, confidential
/// computing): they may exit non-zero, but must report no "Unknown Error".
pub const NVIDIA_SMI_HARDWARE_VIEWS: &[&str] = &[
    "nvidia-smi nvlink -s",
    "nvidia-smi nvlink -c",
    "nvidia-smi nvlink -e",
    "nvidia-smi nvlink -R",
    "nvidia-smi nvlink -p",
    "nvidia-smi mig -lgip",
    "nvidia-smi conf-compute -grs",
];

// Queries every --query-gpu property, one per call; see the file.
const NVIDIA_SMI_SWEEP: &str = include_str!("nvidia/nvidia-smi-sweep.sh");

/// How nvidia-smi reported one property of one GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvidiaSmiValue {
    Value,
    NotAvailable,
    NotSupported,
    InsufficientPermissions,
    // A property the driver no longer reports.
    Deprecated,
    UnknownError,
    // Any other bracketed report, such as "[GPU is lost]".
    OtherError,
    Empty,
    // nvidia-smi rejected the query itself (non-zero exit).
    QueryFailed,
}

impl NvidiaSmiValue {
    pub fn classify(value: &str) -> Self {
        match value.trim() {
            "" => Self::Empty,
            "[N/A]" | "N/A" => Self::NotAvailable,
            "[Not Supported]" => Self::NotSupported,
            "[Insufficient Permissions]" => Self::InsufficientPermissions,
            "[Unknown Error]" => Self::UnknownError,
            "[Requested functionality has been deprecated]" => Self::Deprecated,
            // nvidia-smi brackets what it reports instead of a value; "[GPU is
            // lost]" and "[GPU requires reset]" are errors as well.
            value if value.starts_with('[') && value.ends_with(']') => Self::OtherError,
            _ => Self::Value,
        }
    }

    pub fn is_error(self) -> bool {
        matches!(
            self,
            Self::UnknownError | Self::OtherError | Self::QueryFailed
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaSmiProperty {
    pub property: String,
    // nvidia-smi GPU index, or "-" when the query failed.
    pub gpu: String,
    pub value: String,
    pub class: NvidiaSmiValue,
}

/// The result of `NVIDIA_SMI_SWEEP`: how many properties --help-query-gpu
/// listed, and one entry per property and GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaSmiSweep {
    pub listed: usize,
    pub properties: Vec<NvidiaSmiProperty>,
}

fn parse_nvidia_smi_sweep(output: &str) -> NvidiaSmiSweep {
    let listed = output
        .lines()
        .find_map(|line| line.strip_prefix("CH_SWEEP_LISTED\t"))
        .and_then(|count| count.trim().parse().ok())
        .unwrap_or_default();
    let properties = output
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(4, '\t');
            match (fields.next()?, fields.next()?, fields.next()?) {
                ("CH_SWEEP", property, gpu) => {
                    let value = fields.next().unwrap_or_default().trim().to_string();
                    Some(NvidiaSmiProperty {
                        property: property.to_string(),
                        gpu: gpu.trim().to_string(),
                        class: NvidiaSmiValue::classify(&value),
                        value,
                    })
                }
                ("CH_SWEEP_FAILED", property, message) => Some(NvidiaSmiProperty {
                    property: property.to_string(),
                    gpu: "-".to_string(),
                    value: message.trim().to_string(),
                    class: NvidiaSmiValue::QueryFailed,
                }),
                _ => None,
            }
        })
        .collect();
    NvidiaSmiSweep { listed, properties }
}

impl Guest {
    /// Runs each command; a non-zero exit code or an "Unknown Error" line is
    /// a problem.
    pub fn nvidia_smi_problems(&self, commands: &[&str]) -> Vec<String> {
        self.nvidia_smi_check(commands, true)
    }

    /// Like [`Guest::nvidia_smi_problems`] for views of hardware not every
    /// GPU has: a non-zero exit code is fine, an "Unknown Error" line is not.
    pub fn nvidia_smi_field_errors(&self, commands: &[&str]) -> Vec<String> {
        self.nvidia_smi_check(commands, false)
    }

    fn nvidia_smi_check(&self, commands: &[&str], require_success: bool) -> Vec<String> {
        let mut problems = Vec::new();

        for command in commands {
            let output = match self.ssh_command_exit_code(command) {
                Ok((output, 0)) => output,
                Ok((output, exit_code)) => {
                    if require_success {
                        problems.push(format!(
                            "{command}: exit code {exit_code}: {}",
                            output.lines().next().unwrap_or_default()
                        ));
                    }
                    output
                }
                Err(e) => {
                    problems.push(format!("{command}: {e}"));
                    continue;
                }
            };

            for line in nvidia_smi_error_lines(&output) {
                problems.push(format!("{command}: {}", line.trim()));
            }
        }

        problems
    }

    /// Queries every property nvidia-smi lists in --help-query-gpu, one
    /// property per call, for every GPU in the guest.
    pub fn nvidia_smi_sweep(&self) -> Result<NvidiaSmiSweep, SshCommandError> {
        let (output, _) = self.ssh_command_exit_code(NVIDIA_SMI_SWEEP)?;
        Ok(parse_nvidia_smi_sweep(&output))
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;
    use std::{env, fs};

    use vmm_sys_util::tempdir::TempDir;

    use super::*;
    use crate::{guest_command_with_exit_code, split_guest_exit_code};

    #[test]
    fn nvidia_smi_errors_are_found_in_csv_and_q_output() {
        let broken_csv = "pci.bus_id, pcie.link.gen.current, pcie.link.width.current\n\
                          00000008:00:00.0, 1, [Unknown Error]\n";
        let fixed_csv = "pci.bus_id, pcie.link.gen.current, pcie.link.width.current\n\
                         00000008:00:00.0, 4, 1\n";
        let broken_q = "        GPU Link Info\n\
                        \x20           Link Width\n\
                        \x20               Max                       : Unknown Error\n\
                        \x20               Current                   : Unknown Error\n";
        let ordinary_q = "    Fan Speed                             : N/A\n\
                          \x20   Power Smoothing                       : Insufficient Permissions\n";

        assert_eq!(nvidia_smi_error_lines(broken_csv).len(), 1);
        assert!(nvidia_smi_error_lines(fixed_csv).is_empty());
        assert_eq!(nvidia_smi_error_lines(broken_q).len(), 2);
        assert!(nvidia_smi_error_lines(ordinary_q).is_empty());
    }

    #[test]
    fn nvidia_smi_values_are_classified() {
        assert_eq!(NvidiaSmiValue::classify(" 4"), NvidiaSmiValue::Value);
        assert_eq!(
            NvidiaSmiValue::classify("[N/A]"),
            NvidiaSmiValue::NotAvailable
        );
        assert_eq!(
            NvidiaSmiValue::classify("[Not Supported]"),
            NvidiaSmiValue::NotSupported
        );
        assert_eq!(
            NvidiaSmiValue::classify("[Insufficient Permissions]"),
            NvidiaSmiValue::InsufficientPermissions
        );
        assert_eq!(
            NvidiaSmiValue::classify("[Unknown Error]"),
            NvidiaSmiValue::UnknownError
        );
        assert_eq!(NvidiaSmiValue::classify(" "), NvidiaSmiValue::Empty);
        // Any other bracketed report is an error; a value is not bracketed.
        assert_eq!(
            NvidiaSmiValue::classify("[GPU is lost]"),
            NvidiaSmiValue::OtherError
        );
        assert_eq!(
            NvidiaSmiValue::classify("NVIDIA GH200 480GB"),
            NvidiaSmiValue::Value
        );

        assert!(NvidiaSmiValue::UnknownError.is_error());
        assert!(NvidiaSmiValue::OtherError.is_error());
        assert!(NvidiaSmiValue::QueryFailed.is_error());
        assert!(!NvidiaSmiValue::Empty.is_error());
        assert_eq!(
            NvidiaSmiValue::classify("[Requested functionality has been deprecated]"),
            NvidiaSmiValue::Deprecated
        );
        assert!(!NvidiaSmiValue::Deprecated.is_error());
        assert!(!NvidiaSmiValue::NotAvailable.is_error());
        assert!(!NvidiaSmiValue::InsufficientPermissions.is_error());
    }

    // Runs the sweep script itself against a fake nvidia-smi with two GPUs:
    #[test]
    fn nvidia_smi_sweep_script_runs_one_property_per_query() {
        let dir = TempDir::new_with_prefix("/tmp/ch-nvsmi").unwrap();
        let fake = dir.as_path().join("nvidia-smi");
        fs::write(
            &fake,
            r#"#!/bin/bash
if [ "$1" = "--help-query-gpu" ]; then
    printf '%s\n' 'List of valid properties to query for the switch "--query-gpu=":' \
        '"pcie.link.gen.current" or "pcie.link.gen.gpucurrent"' 'The current link generation.' \
        '"pcie.link.width.current"' '"clocks_event_reasons.active"' '"fan.speed"' \
        '"gone.property"' '"pcie.link.gen.current"'
    exit 0
fi
case "$1" in
    *pcie.link.gen.current*) printf '0, 4\n1, 4\n' ;;
    *pcie.link.width.current*) printf '0, [Unknown Error]\n1, 1\n' ;;
    *clocks_event_reasons.active*) printf '0, 0x1, extra\n1, 0x1\n' ;;
    *fan.speed*) printf '0, [N/A]\n1, [N/A]\n' ;;
    *) echo 'Field "gone.property" is not a valid field to query.'; exit 2 ;;
esac
"#,
        )
        .unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();

        let output = Command::new("bash")
            .arg("-c")
            .arg(guest_command_with_exit_code(NVIDIA_SMI_SWEEP))
            .env(
                "PATH",
                format!("{}:{}", dir.as_path().display(), env::var("PATH").unwrap()),
            )
            .output()
            .unwrap();
        let output = String::from_utf8(output.stdout).unwrap();
        let (sweep_output, exit_code) = split_guest_exit_code(&output);
        assert_eq!(exit_code, Some(0), "{output}");

        let sweep = parse_nvidia_smi_sweep(sweep_output);
        assert_eq!(sweep.listed, 5, "{sweep:#?}");
        let rows = sweep.properties;
        let find = |property: &str, gpu: &str| {
            rows.iter()
                .find(|r| r.property == property && r.gpu == gpu)
                .unwrap_or_else(|| panic!("no row for {property} on GPU {gpu}: {rows:#?}"))
        };

        assert_eq!(rows.len(), 9, "{rows:#?}");
        assert_eq!(find("pcie.link.gen.current", "1").value, "4");
        assert_eq!(
            find("pcie.link.width.current", "0").class,
            NvidiaSmiValue::UnknownError
        );
        assert_eq!(find("pcie.link.width.current", "1").value, "1");
        assert_eq!(find("clocks_event_reasons.active", "0").value, "0x1, extra");
        assert_eq!(find("fan.speed", "0").class, NvidiaSmiValue::NotAvailable);
        let failed = find("gone.property", "-");
        assert_eq!(failed.class, NvidiaSmiValue::QueryFailed);
        assert!(failed.value.contains("not a valid field"));
        assert_eq!(rows.iter().filter(|r| r.class.is_error()).count(), 2);
    }
}
