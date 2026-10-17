// Copyright (c) 2026 Crusoe Energy Systems LLC
//
// SPDX-License-Identifier: Apache-2.0
//

//! Checks of an NVIDIA GPU passed through to a guest: nvidia-smi views and
//! properties, DCGM's diagnostic and tools, and the GPU's fault counters.

use std::time::{Duration, Instant};
use std::{fmt, thread};

use serde_json::Value;

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

/// The status DCGM gives a diagnostic test, in its summary and per entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DcgmDiagStatus {
    Pass,
    Fail,
    Skip,
    Warn,
    /// A status this parser does not know, as DCGM printed it.
    Other(String),
}

impl DcgmDiagStatus {
    fn parse(status: &str) -> Self {
        match status {
            "Pass" => Self::Pass,
            "Fail" => Self::Fail,
            "Skip" => Self::Skip,
            "Warn" => Self::Warn,
            other => Self::Other(other.to_string()),
        }
    }
}

impl fmt::Display for DcgmDiagStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => f.write_str("Pass"),
            Self::Fail => f.write_str("Fail"),
            Self::Skip => f.write_str("Skip"),
            Self::Warn => f.write_str("Warn"),
            Self::Other(status) => f.write_str(status),
        }
    }
}

/// One test of a "dcgmi diag -j" run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DcgmDiagTest {
    pub category: String,
    pub name: String,
    // From the test's summary.
    pub status: DcgmDiagStatus,
    // The status of each entity (a GPU, in a guest) the test ran on, by ID.
    pub results: Vec<(u64, DcgmDiagStatus)>,
    // Warnings DCGM attached to the test's per-GPU results.
    pub messages: Vec<String>,
}

impl DcgmDiagTest {
    pub fn failed(&self) -> bool {
        self.status == DcgmDiagStatus::Fail
            || self
                .results
                .iter()
                .any(|(_, status)| *status == DcgmDiagStatus::Fail)
    }

    /// The test ran on at least one GPU and passed on every one: a summary
    /// "Pass" over a GPU the test skipped is not a pass.
    pub fn passed(&self) -> bool {
        self.status == DcgmDiagStatus::Pass
            && !self.results.is_empty()
            && self
                .results
                .iter()
                .all(|(_, status)| *status == DcgmDiagStatus::Pass)
    }
}

/// Parses the JSON that "dcgmi diag -j" prints (DCGM 4.x).
fn parse_dcgm_diag(output: &str) -> Result<Vec<DcgmDiagTest>, String> {
    // The JSON starts on a line of its own; stderr, merged into the output,
    // may hold a brace before it.
    let json = output
        .lines()
        .scan(0, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find(|(_, line)| line.trim_start().starts_with('{'))
        .map(|(at, _)| &output[at..])
        .ok_or_else(|| format!("no JSON in dcgmi output: {output}"))?;
    // stderr is merged into the output, so only the first JSON value is read
    // and whatever follows it is left alone.
    let json: Value = serde_json::Deserializer::from_str(json)
        .into_iter::<Value>()
        .next()
        .ok_or_else(|| format!("no JSON in dcgmi output: {output}"))?
        .map_err(|e| format!("bad dcgmi JSON ({e}): {output}"))?;
    let categories = json["DCGM Diagnostic"]["test_categories"]
        .as_array()
        .ok_or_else(|| format!("no test categories in dcgmi JSON: {output}"))?;

    let mut tests = Vec::new();
    for category in categories {
        for test in category["tests"].as_array().into_iter().flatten() {
            let results = test["results"].as_array().into_iter().flatten();
            let statuses = results
                .clone()
                .map(|result| {
                    (
                        result["entity_id"].as_u64().unwrap_or_default(),
                        DcgmDiagStatus::parse(result["status"].as_str().unwrap_or_default()),
                    )
                })
                .collect();
            let messages = results
                .flat_map(|result| {
                    let entity = result["entity_id"].as_u64().unwrap_or_default();
                    result["warnings"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(move |w| {
                            Some(format!("GPU {entity}: {}", w["warning"].as_str()?))
                        })
                })
                .collect();
            tests.push(DcgmDiagTest {
                category: category["category"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                name: test["name"].as_str().unwrap_or_default().to_string(),
                status: DcgmDiagStatus::parse(
                    test["test_summary"]["status"].as_str().unwrap_or_default(),
                ),
                results: statuses,
                messages,
            });
        }
    }

    Ok(tests)
}

/// The count in a "dcgmi discovery -l" heading for `entity`: "4 GPUs found."
/// or, since DCGM 4.7, "1 GPU found (Active).".
fn dcgmi_found_count(line: &str, entity: &str) -> Option<u32> {
    let (count, rest) = line.trim().split_once(' ')?;
    let rest = rest.strip_prefix(entity)?;
    let rest = rest.strip_prefix('s').unwrap_or(rest);
    if !rest.starts_with(" found") {
        return None;
    }
    count.parse().ok()
}

/// GPU IDs listed by "dcgmi discovery -l".
pub fn dcgmi_gpu_ids(output: &str) -> Vec<u32> {
    output
        .lines()
        .skip_while(|line| dcgmi_found_count(line, "GPU").is_none())
        .skip(1)
        .take_while(|line| !line.contains(" found"))
        .filter_map(|line| line.split('|').nth(1)?.trim().parse().ok())
        .collect()
}

/// The group ID that "dcgmi group -c" or "dcgmi fieldgroup -c" printed
/// ("... with a group ID of 3").
pub fn dcgmi_created_id(output: &str) -> Option<u32> {
    let (_, id) = output.split_once("ID of ")?;
    id.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

/// The number of NVIDIA CPUs (Grace) that "dcgmi discovery -l" lists.
pub fn dcgmi_cpu_count(output: &str) -> u32 {
    output
        .lines()
        .find_map(|line| dcgmi_found_count(line, "CPU"))
        .unwrap_or_default()
}

/// The "Current" column of `setting` in "dcgmi config --get" output, e.g.
/// "Unrestricted" for "Compute Mode"; "****" when the GPUs of the group
/// differ.
pub fn dcgmi_config_current(output: &str, setting: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let columns = line.split('|').collect::<Vec<_>>();
        if columns.len() < 5 || !columns[1].trim().eq_ignore_ascii_case(setting) {
            return None;
        }
        Some(columns[columns.len() - 2].trim().to_string())
    })
}

/// The (min, max) power limit in watts from "dcgmi discovery -i p".
pub fn dcgmi_power_range(output: &str) -> Option<(u32, u32)> {
    let limit = |name: &str| {
        output.lines().find_map(|line| {
            let mut columns = line.split('|').skip(1);
            if !columns.next()?.trim().starts_with(name) {
                return None;
            }
            columns.next()?.trim().parse::<u32>().ok()
        })
    };
    let (min, max) = (limit("Min Power Limit")?, limit("Max Power Limit")?);
    (min <= max).then_some((min, max))
}

// Samples every field of "dcgmi dmon -l" once, one per call; see the file.
const DCGM_DMON_SWEEP: &str = include_str!("nvidia/dcgm-dmon-sweep.sh");

/// How `DCGM_DMON_SWEEP` fared with one field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DcgmFieldResult {
    /// dcgmi printed a sample row for the entity; the value may be "N/A".
    Sampled,
    /// dcgmi exited 0 but printed no sample row for any entity.
    NoSample,
    /// dcgmi could not watch the field; the value is its first line.
    Failed,
}

/// One field of one entity in a `DCGM_DMON_SWEEP`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DcgmFieldValue {
    pub id: String,
    pub name: String,
    // "GPU 0", or "-" when the field was not sampled.
    pub entity: String,
    // The sampled value ("N/A" when DCGM has none), dcgmi's error, or
    // nothing.
    pub value: String,
    pub result: DcgmFieldResult,
}

/// The result of `DCGM_DMON_SWEEP`: how many fields were listed, and one
/// entry per field and entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DcgmDmonSweep {
    pub listed: usize,
    pub fields: Vec<DcgmFieldValue>,
}

fn parse_dcgm_dmon_sweep(output: &str) -> DcgmDmonSweep {
    let listed = output
        .lines()
        .find_map(|line| line.strip_prefix("CH_DMON_LISTED\t"))
        .and_then(|count| count.trim().parse().ok())
        .unwrap_or_default();
    let fields = output
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(5, '\t');
            let (tag, id, name) = (fields.next()?, fields.next()?, fields.next()?);
            let (entity, value, result) = match tag {
                "CH_DMON" => (
                    fields.next()?,
                    fields.next().unwrap_or_default(),
                    DcgmFieldResult::Sampled,
                ),
                "CH_DMON_NOSAMPLE" => ("-", "", DcgmFieldResult::NoSample),
                "CH_DMON_FAILED" => (
                    "-",
                    fields.next().unwrap_or_default(),
                    DcgmFieldResult::Failed,
                ),
                _ => return None,
            };
            Some(DcgmFieldValue {
                id: id.to_string(),
                name: name.to_string(),
                entity: entity.to_string(),
                value: value.trim().to_string(),
                result,
            })
        })
        .collect();
    DcgmDmonSweep { listed, fields }
}

fn gpu_kernel_error_lines(dmesg: &str) -> Vec<&str> {
    // "Correctable" and "Uncorrectable" since kernel 6.8, "Corrected" and
    // "Uncorrected" before.
    const ERRORS: &[&str] = &[
        "NVRM: Xid",
        "AER: Correctable",
        "AER: Corrected",
        "AER: Uncorrectable",
        "AER: Uncorrected",
        "AER: Multiple",
        "DPC: containment",
        "DPC: unmasked",
    ];
    dmesg
        .lines()
        .filter(|line| ERRORS.iter().any(|error| line.contains(error)))
        .collect()
}

const GPU_ERROR_COUNTERS: &str = include_str!("nvidia/gpu-error-counters.sh");

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

    /// Whether the guest has dcgmi. "command -v" exits 1 (127 under dash)
    /// when it does not; anything else is a failure of the guest or of the
    /// connection, not a missing DCGM.
    pub fn dcgm_installed(&self) -> Result<bool, String> {
        match self.ssh_command_exit_code("command -v dcgmi") {
            Ok((_, 0)) => Ok(true),
            Ok((_, 1 | 127)) => Ok(false),
            Ok((output, exit_code)) => Err(format!(
                "command -v dcgmi exited with {exit_code}: {output}"
            )),
            Err(e) => Err(format!("command -v dcgmi: {e}")),
        }
    }

    /// Starts DCGM's host engine (the packaged service if there is one) and
    /// waits up to 30 s for it to list every one of the `gpus` passed
    /// through. The error carries what the start and the last listing said.
    pub fn start_dcgm(&self, gpus: usize) -> Result<(), String> {
        let (started, _) = self
            .ssh_command_exit_code("sudo systemctl start nvidia-dcgm || sudo nv-hostengine")
            .map_err(|e| format!("starting the host engine: {e}"))?;

        let start = Instant::now();
        let mut listing = String::new();
        while start.elapsed() < Duration::from_secs(30) {
            match self.ssh_command_exit_code("dcgmi discovery -l") {
                Ok((output, 0)) if dcgmi_gpu_ids(&output).len() == gpus => return Ok(()),
                Ok((output, _)) => listing = output,
                Err(e) => listing = format!("ssh failed: {e}"),
            }
            thread::sleep(Duration::from_millis(500));
        }
        Err(format!(
            "DCGM lists {} of {gpus} GPUs after 30 s\nstart: {}\nlast listing: {listing}",
            dcgmi_gpu_ids(&listing).len(),
            started.trim()
        ))
    }

    /// Runs `dcgmi diag <options> -j`. dcgmi exits non-zero when a test fails,
    /// so only a run still going after `timeout_secs` is an error.
    pub fn dcgm_diag(&self, options: &str, timeout_secs: u32) -> Result<Vec<DcgmDiagTest>, String> {
        // The diagnostic prints nothing until it is done, so the connection
        // may be silent for the whole timeout.
        let (output, exit_code) = self
            .ssh_command_exit_code_timeout(
                &format!("sudo timeout --kill-after=60 {timeout_secs} dcgmi diag {options} -j"),
                Duration::from_secs(u64::from(timeout_secs) + 120),
            )
            .map_err(|e| format!("ssh failed: {e}"))?;
        // timeout(1) exits 124 after SIGTERM, 137 after the SIGKILL.
        if exit_code == 124 || exit_code == 137 {
            return Err(format!("timed out after {timeout_secs} s: {output}"));
        }
        // dcgmi killed by a signal prints nothing; the shell's exit code
        // names the signal (132 for SIGILL).
        if output.trim().is_empty() {
            return Err(format!("dcgmi exited with {exit_code} and printed nothing"));
        }
        parse_dcgm_diag(&output)
    }

    /// Samples every field "dcgmi dmon -l" lists once, one per call.
    pub fn dcgm_dmon_sweep(&self) -> Result<DcgmDmonSweep, SshCommandError> {
        let (output, _) = self.ssh_command_exit_code(DCGM_DMON_SWEEP)?;
        Ok(parse_dcgm_dmon_sweep(&output))
    }

    /// The Xid, AER and DPC error lines of the guest kernel log since boot.
    pub fn gpu_kernel_errors(&self) -> Result<Vec<String>, String> {
        let (dmesg, exit_code) = self
            .ssh_command_exit_code("sudo dmesg")
            .map_err(|e| format!("ssh failed: {e}"))?;
        if exit_code != 0 {
            return Err(format!(
                "dmesg exited with {exit_code}: {}",
                dmesg.lines().next().unwrap_or_default()
            ));
        }
        Ok(gpu_kernel_error_lines(&dmesg)
            .into_iter()
            .map(str::to_string)
            .collect())
    }

    /// The GPUs' error counters, one line each, named by GPU and section; a
    /// line that appears between two readings is a counter that moved.
    pub fn gpu_error_counters(&self) -> Result<Vec<String>, String> {
        let (output, exit_code) = self
            .ssh_command_exit_code(GPU_ERROR_COUNTERS)
            .map_err(|e| format!("ssh failed: {e}"))?;
        if exit_code != 0 {
            return Err(format!(
                "the counter script exited with {exit_code}: {}",
                output.lines().last().unwrap_or_default()
            ));
        }
        Ok(output.lines().map(str::to_string).collect())
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

    // Shortened from real "dcgmi diag -j" output (DCGM 4.4.2, GB200 guests):
    // a passing level 1 run, and a level 3 run whose nvbandwidth test failed.
    const DCGM_DIAG_PASS: &str = r#"{"DCGM Diagnostic": {"test_categories": [
        {"category": "Deployment", "tests": [{"name": "software",
         "results": [{"entity_group": "GPU", "entity_group_id": 1, "entity_id": 0, "status": "Pass"},
                     {"entity_group": "GPU", "entity_group_id": 1, "entity_id": 1, "status": "Pass"}],
         "test_summary": {"status": "Pass"}}]}]},
        "metadata": {"Driver Version Detected": "580.95.05", "version": "4.4.2"}}"#;
    const DCGM_DIAG_FAIL: &str = r#"{"DCGM Diagnostic": {"test_categories": [
        {"category": "Hardware", "tests": [
         {"name": "memory", "results": [{"entity_id": 0, "status": "Pass"}], "test_summary": {"status": "Pass"}},
         {"name": "nvbandwidth", "results": [
            {"entity_id": 0, "status": "Fail", "warnings": [{"error_category": 10, "error_id": 38,
             "error_severity": 2, "warning": "The memory copy utilization for the GPU: 0 is greater than 10%."}]},
            {"entity_id": 1, "status": "Fail"}],
          "test_summary": {"status": "Fail"}}]},
        {"category": "Integration", "tests": [
         {"name": "pcie", "results": [{"entity_id": 0, "status": "Skip"}], "test_summary": {"status": "Skip"}}]}]}}"#;

    #[test]
    fn dcgm_diag_results_are_parsed() {
        let tests = parse_dcgm_diag(DCGM_DIAG_PASS).unwrap();
        assert_eq!(
            tests,
            vec![DcgmDiagTest {
                category: "Deployment".to_string(),
                name: "software".to_string(),
                status: DcgmDiagStatus::Pass,
                results: vec![(0, DcgmDiagStatus::Pass), (1, DcgmDiagStatus::Pass)],
                messages: vec![],
            }]
        );
        assert!(!tests[0].failed());
        assert!(tests[0].passed());

        // Text before the JSON (a warning from dcgmi) is skipped.
        let tests = parse_dcgm_diag(&format!("Warning: foo\n{DCGM_DIAG_FAIL}")).unwrap();
        let failed = tests.iter().filter(|t| t.failed()).collect::<Vec<_>>();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].name, "nvbandwidth");
        assert_eq!(failed[0].category, "Hardware");
        assert_eq!(
            failed[0].messages,
            vec!["GPU 0: The memory copy utilization for the GPU: 0 is greater than 10%."]
        );
        // A skipped test is reported but is not a failure.
        let pcie = tests.iter().find(|t| t.name == "pcie").unwrap();
        assert_eq!(pcie.status, DcgmDiagStatus::Skip);
        assert!(!pcie.failed());
        assert!(!pcie.passed());

        // Shortened from "dcgmi diag -r 2 -j" in a GB200 guest with four GPUs
        // (DCGM 4.4.2): the summary and every GPU pass.
        let four_gpus = r#"{"DCGM Diagnostic": {"test_categories": [
            {"category": "Hardware", "tests": [{"name": "memory", "results": [
                {"entity_group": "GPU", "entity_group_id": 1, "entity_id": 0, "status": "Pass",
                 "info": ["Allocated 194863357557 bytes (98.6%)"]},
                {"entity_group": "GPU", "entity_group_id": 1, "entity_id": 1, "status": "Pass"},
                {"entity_group": "GPU", "entity_group_id": 1, "entity_id": 2, "status": "Pass"},
                {"entity_group": "GPU", "entity_group_id": 1, "entity_id": 3, "status": "Pass"}],
             "test_summary": {"status": "Pass"}}]}]},
            "metadata": {"Driver Version Detected": "580.95.05", "version": "4.4.2"}}"#;
        let tests = parse_dcgm_diag(four_gpus).unwrap();
        assert_eq!(tests[0].results.len(), 4, "{tests:#?}");
        assert!(tests[0].passed());

        // A summary "Pass" over a GPU the test skipped is not a pass.
        let one_skipped = four_gpus.replace(
            r#""entity_id": 2, "status": "Pass""#,
            r#""entity_id": 2, "status": "Skip""#,
        );
        let tests = parse_dcgm_diag(&one_skipped).unwrap();
        assert_eq!(tests[0].status, DcgmDiagStatus::Pass);
        assert!(!tests[0].passed());
        assert!(!tests[0].failed());

        let error = parse_dcgm_diag("Error: Unable to connect to host engine").unwrap_err();
        assert!(error.contains("no JSON"), "{error}");
        let error = parse_dcgm_diag("{\"unexpected\": 1}").unwrap_err();
        assert!(error.contains("no test categories"), "{error}");
    }

    // Shortened from real dcgmi 4.4.2 output in a GB200 guest.
    const DCGMI_DISCOVERY: &str = "4 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA GB200                                                   |
|        | PCI Bus ID: 00000008:00:00.0                                         |
+--------+----------------------------------------------------------------------+
| 3      | Name: NVIDIA GB200                                                   |
|        | PCI Bus ID: 0000000B:00:00.0                                         |
+--------+----------------------------------------------------------------------+
0 NvSwitches found.
+-----------+
| Switch ID |
+-----------+
+-----------+
2 CPUs found.
+--------+----------------------------------------------------------------------+
| CPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: Grace TH500                                                    |
| 1      | Name: Grace TH500                                                    |
+--------+----------------------------------------------------------------------+
";

    #[test]
    fn dcgmi_gpu_ids_are_parsed() {
        assert_eq!(dcgmi_gpu_ids(DCGMI_DISCOVERY), vec![0, 3]);
        let one_gpu = "1 GPU found.\n| 0      | Name: NVIDIA L40S |\n0 CPUs found.\n";
        assert_eq!(dcgmi_gpu_ids(one_gpu), vec![0]);
        // DCGM 4.7 in a GH200 guest qualifies the GPU heading.
        let dcgm_4_7 = "1 GPU found (Active).\n\
                        | 0      | Name: NVIDIA GH200 480GB |\n\
                        0 NvSwitches found.\n\
                        0 CPUs found.\n";
        assert_eq!(dcgmi_gpu_ids(dcgm_4_7), vec![0]);
        // Ten GPUs: the heading contains "0 GPUs".
        let ten_gpus = format!(
            "10 GPUs found.\n{}0 NvSwitches found.\n",
            (0..10)
                .map(|gpu| format!("| {gpu}      | Name: NVIDIA GB200 |\n"))
                .collect::<String>()
        );
        assert_eq!(dcgmi_gpu_ids(&ten_gpus), (0..10).collect::<Vec<_>>());
        assert!(dcgmi_gpu_ids("0 GPUs found.\n0 NvSwitches found.\n").is_empty());
        assert!(dcgmi_gpu_ids("Error: unable to connect").is_empty());
    }

    // Shortened from real dcgmi 4.4.2 output in a GB200 guest.
    const DCGMI_CONFIG: &str = "\
+------------------------------+------------------------------+------------------------------+
| dcgmi_test_group                                                                           |
| Group of 4 GPUs                                                                            |
+==============================+==============================+==============================+
| Field                        | Target                       | Current                      |
+------------------------------+------------------------------+------------------------------+
| Compute Mode                 | Not Specified                | Unrestricted                 |
| ECC Mode                     | Not Specified                | Enabled                      |
| Sync Boost                   | Not Specified                | Not Supported                |
| Power Limit                  | Not Specified                | 1200                         |
+------------------------------+------------------------------+------------------------------+
**** Non-homogenous settings across group. Use with -v flag to see details.
";
    const DCGMI_POWER: &str = "\
+--------------------------+-------------------------------------------------+
| GPU ID: 0                | Device Information                              |
+==========================+=================================================+
| Current Power Limit (W)  | 1200                                            |
| Default Power Limit (W)  | 1200                                            |
| Max Power Limit (W)      | 1200                                            |
| Min Power Limit (W)      | 200                                             |
+--------------------------+-------------------------------------------------+
";

    #[test]
    fn dcgmi_output_is_parsed() {
        assert_eq!(dcgmi_cpu_count(DCGMI_DISCOVERY), 2);
        assert_eq!(dcgmi_cpu_count("1 GPU found (Active).\n0 CPUs found.\n"), 0);
        assert_eq!(dcgmi_cpu_count("1 GPU found.\n"), 0);

        assert_eq!(
            dcgmi_created_id(
                "Successfully created group \"ch_dcgm_group\" with a group ID of 19\n"
            ),
            Some(19)
        );
        assert_eq!(
            dcgmi_created_id("Successfully created field group \"f\" with a field group ID of 492"),
            Some(492)
        );
        assert_eq!(dcgmi_created_id("Error: group exists"), None);

        assert_eq!(
            dcgmi_config_current(DCGMI_CONFIG, "Compute Mode").as_deref(),
            Some("Unrestricted")
        );
        assert_eq!(
            dcgmi_config_current(DCGMI_CONFIG, "sync boost").as_deref(),
            Some("Not Supported")
        );
        assert_eq!(
            dcgmi_config_current(DCGMI_CONFIG, "Power Limit").as_deref(),
            Some("1200")
        );
        assert_eq!(dcgmi_config_current(DCGMI_CONFIG, "Power Profile"), None);

        assert_eq!(dcgmi_power_range(DCGMI_POWER), Some((200, 1200)));
        assert_eq!(dcgmi_power_range(DCGMI_CONFIG), None);
    }

    // Runs the dmon sweep script itself against a fake dcgmi with two GPUs.
    #[test]
    fn dcgm_dmon_sweep_script_runs_one_field_per_call() {
        let dir = TempDir::new_with_prefix("/tmp/ch-dcgmi").unwrap();
        let fake = dir.as_path().join("dcgmi");
        fs::write(
            &fake,
            r#"#!/bin/bash
if [ "$1 $2" = "dmon -l" ]; then
    printf '%s\n' '___________________________________________' \
        'Long Name                 Short Name          Field ID' \
        '___________________________________________' \
        'name                      DVNAM               50' \
        'gpu_temp                  TMPTR               150' \
        'fan_speed                 FANSP               160' \
        'pcie_tx_throughput        TXPCI               200' \
        'pcie_link_width           PCIWD               238' \
        'header_only               HDRON               300' \
        'gone_field                GONE                999' \
        'gpu_temp                  TMPTR               150'
    exit 0
fi
case "$3" in
    50) printf '#Entity   DVNAM\nID\nGPU 0     NVIDIA GB200\nGPU 1     NVIDIA GB200\n' ;;
    150) printf '#Entity   TMPTR\nID        C\nGPU 0     33\nGPU 1     34\n' ;;
    160) echo '  Error: Unable to watch field 160' ;;
    238) printf '#Entity   PCIWD\nID\nGPU 0     N/A\nGPU 1     16\n' ;;
    300) printf '#Entity   HDRON\nID\n' ;;
    *) echo "Error: field $3 queried"; exit 1 ;;
esac
"#,
        )
        .unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();

        let output = Command::new("bash")
            .arg("-c")
            .arg(guest_command_with_exit_code(DCGM_DMON_SWEEP))
            .env(
                "PATH",
                format!("{}:{}", dir.as_path().display(), env::var("PATH").unwrap()),
            )
            .output()
            .unwrap();
        let output = String::from_utf8(output.stdout).unwrap();
        let (sweep_output, exit_code) = split_guest_exit_code(&output);
        assert_eq!(exit_code, Some(0), "{output}");

        let sweep = parse_dcgm_dmon_sweep(sweep_output);
        assert_eq!(sweep.listed, 6, "{sweep:#?}");
        let rows = sweep.fields;
        let find = |id: &str, entity: &str| {
            rows.iter()
                .find(|r| r.id == id && r.entity == entity)
                .unwrap_or_else(|| panic!("no row for field {id} of {entity}: {rows:#?}"))
        };

        assert_eq!(rows.len(), 9, "{rows:#?}");
        assert_eq!(find("300", "-").result, DcgmFieldResult::NoSample);
        assert_eq!(find("300", "-").value, "");
        let mut ids = rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), sweep.listed, "{rows:#?}");
        assert_eq!(find("50", "GPU 1").value, "NVIDIA GB200");
        assert_eq!(find("150", "GPU 0").value, "33");
        assert_eq!(find("150", "GPU 0").name, "gpu_temp");
        assert_eq!(find("238", "GPU 0").value, "N/A");
        assert_eq!(find("238", "GPU 0").result, DcgmFieldResult::Sampled);
        assert_eq!(find("160", "-").result, DcgmFieldResult::Failed);
        assert!(find("999", "-").value.contains("field 999 queried"));
        assert!(!rows.iter().any(|r| r.id == "200"), "{rows:#?}");
        assert_eq!(
            rows.iter()
                .filter(|r| r.result == DcgmFieldResult::Failed)
                .count(),
            2
        );
    }

    #[test]
    fn gpu_kernel_errors_are_found() {
        let dmesg = "\
[    5.1] nvidia: loading out-of-tree module taints kernel.
[    6.2] NVRM: loading NVIDIA UNIX Open Kernel Module for aarch64  580.95.05
[    7.0] pcieport 0000:00:01.0: AER: enabled with IRQ 24
[    7.1] pcieport 0000:00:01.0: DPC: enabled with IRQ 24
[  120.3] NVRM: Xid (PCI:0008:00:00): 79, pid=1234, GPU has fallen off the bus.
[  121.0] pcieport 0000:00:01.0: AER: Correctable error message received from 0000:01:00.0
[  121.5] pcieport 0000:00:01.0: AER: Multiple Uncorrectable (Non-Fatal) error message received from 0000:01:00.0
[  121.7] pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: 0000:01:00.0
[  122.0] pcieport 0000:00:01.0: DPC: containment event, status:0x1f01
";
        let errors = gpu_kernel_error_lines(dmesg);
        assert_eq!(errors.len(), 5, "{errors:#?}");
        assert!(errors[0].contains("Xid"));
        assert!(errors[1].contains("Correctable"));
    }

    #[test]
    fn gpu_error_counters_name_their_gpu_and_section() {
        let dir = TempDir::new_with_prefix("/tmp/ch-nvsmi-q").unwrap();
        let fake = dir.as_path().join("nvidia-smi");
        fs::write(
            &fake,
            r#"#!/bin/bash
for gpu in 00000008:00:00.0 00000009:00:00.0; do
    volatile=0
    aggregate=0
    [ "$gpu" = 00000008:00:00.0 ] && volatile=1
    [ "$gpu" = 00000009:00:00.0 ] && aggregate=1
    [ "$gpu" = 00000009:00:00.0 ] && [ -n "$CH_AFTER" ] && volatile=1
    printf '%s\n' "GPU $gpu" \
        '    Product Name                          : NVIDIA GB200' \
        '    PCI' \
        '        Replays Since Reset               : 0' \
        '    ECC Errors' \
        '        Volatile' \
        "            SRAM Correctable              : $volatile" \
        '        Aggregate' \
        "            SRAM Correctable              : $aggregate" \
        '    Remapped Rows' \
        '        Pending                           : No' \
        ''
done
"#,
        )
        .unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();

        let counters = |after: bool| {
            let mut command = Command::new("bash");
            command.arg("-c").arg(GPU_ERROR_COUNTERS).env(
                "PATH",
                format!("{}:{}", dir.as_path().display(), env::var("PATH").unwrap()),
            );
            if after {
                command.env("CH_AFTER", "1");
            }
            let output = command.output().unwrap();
            // Only the nvidia-smi lines: a host with an NVIDIA GPU also
            // reports its AER counters.
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .filter(|line| line.contains(" / "))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };

        let before = counters(false);
        assert_eq!(before.len(), 8, "{before:#?}");
        let mut unique = before.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), before.len(), "{before:#?}");
        let aggregate = "00000009:00:00.0 / ECC Errors / Aggregate / SRAM Correctable : 1";
        assert!(before.iter().any(|line| line == aggregate), "{before:#?}");
        assert!(!before.iter().any(|line| line.contains("Product Name")));

        let changed = counters(true)
            .into_iter()
            .filter(|line| !before.contains(line))
            .collect::<Vec<_>>();
        assert_eq!(
            changed,
            vec!["00000009:00:00.0 / ECC Errors / Volatile / SRAM Correctable : 1".to_string()]
        );
    }
}
