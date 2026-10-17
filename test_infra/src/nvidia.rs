// Copyright (c) 2026 Crusoe Energy Systems LLC
//
// SPDX-License-Identifier: Apache-2.0
//

//! Checks of an NVIDIA GPU passed through to a guest: nvidia-smi views.

use crate::Guest;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
