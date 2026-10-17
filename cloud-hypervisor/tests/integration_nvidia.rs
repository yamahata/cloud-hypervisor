// Copyright (c) 2026 Crusoe Energy Systems LLC
//
// SPDX-License-Identifier: Apache-2.0
//
#![cfg(all(target_arch = "aarch64", any(devcli_testenv, clippy)))]

// Experimental tests of a Grace GPU (GH200, GB200, GB300) passed through to
// an aarch64 guest; see #9139. CI has no such runner: run them by hand, as
// root, after a release build, with BUILD_TARGET set:
//   RUSTFLAGS='--cfg devcli_testenv' cargo test -p cloud-hypervisor \
//       --test integration_nvidia -- --ignored --test-threads=1
// Every GPU bound to nvgrace_gpu_vfio_pci is tested; without one the tests
// print SKIPPED. The guest image is NVIDIA_IMAGE_NAME in ~/workloads.
mod nvidia_gpu {
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};
    use std::{fs, panic, thread};

    use test_infra::nvidia::*;
    use test_infra::*;

    // Every Grace GPU of the host bound to GRACE_GPU_DRIVER.
    struct GpuTarget {
        devices: Vec<String>,
    }

    // Every GPU of the host bound to GRACE_GPU_DRIVER, by sysfs path in bus
    // order. A Grace GPU on another driver is in use by the host and skipped.
    fn grace_vfio_gpus() -> Vec<String> {
        let mut gpus = Vec::new();
        for entry in fs::read_dir("/sys/bus/pci/devices")
            .into_iter()
            .flatten()
            .flatten()
        {
            let device = entry.path();
            let driver = fs::read_link(device.join("driver"))
                .ok()
                .and_then(|path| path.file_name().map(|n| n.to_string_lossy().into_owned()));
            if driver.as_deref() == Some(GRACE_GPU_DRIVER) {
                gpus.push(device.to_string_lossy().into_owned());
            }
        }
        gpus.sort();
        gpus
    }

    // Ubuntu 24.04 with Ubuntu's NVIDIA kernel, the open driver and DCGM, built
    // by "IMAGE_ARCH=arm64 NVIDIA_CUSTOM_IMAGE=-nvidia scripts/build-custom-image.sh"
    // and placed in ~/workloads by hand until the CI hosts such an image.
    const NVIDIA_IMAGE_NAME: &str = "noble-server-cloudimg-arm64-custom-nvidia-20261022-0.raw";

    fn nvidia_image_path() -> PathBuf {
        let mut path = dirs::home_dir().unwrap();
        path.push("workloads");
        path.push(NVIDIA_IMAGE_NAME);
        path
    }

    // The vfio-pci variant driver of a Grace GPU, whose memory the guest
    // onlines as NUMA memory.
    const GRACE_GPU_DRIVER: &str = "nvgrace_gpu_vfio_pci";
    // With 48 bits the GPU's memory BAR, over 100 GB, lands above the guest
    // kernel's linear map ("NVRM: Memory add failed").
    const GRACE_MAX_PHYS_BITS: u8 = 47;
    // Memory-less Generic Initiator NUMA nodes per GPU, into which the driver
    // onlines the GPU's memory; without them the GPU fails to start.
    const GRACE_GPU_NUMA_NODES: usize = 8;

    impl GpuTarget {
        fn discover() -> Option<Self> {
            let devices = grace_vfio_gpus();
            if devices.is_empty() {
                println!("SKIPPED: no GPU is bound to {GRACE_GPU_DRIVER}");
                return None;
            }
            let image = nvidia_image_path();
            assert!(
                image.exists(),
                "{} GPUs are bound to {GRACE_GPU_DRIVER} but the guest image {} is missing",
                devices.len(),
                image.display()
            );
            Some(Self { devices })
        }

        fn command<'a>(&self, guest: &'a Guest) -> GuestCommand<'a> {
            let mut command = GuestCommand::new(guest);
            command.args(["--firmware", edk2_path().to_str().unwrap()]);
            // vfio_p2p_dma=on: the GPU reaches its own memory window through
            // the SMMU, and CUDA never creates a context without it.
            let cpus = format!("boot=4,max_phys_bits={GRACE_MAX_PHYS_BITS}");
            let platform = format!(
                "iommufd=on,vfio_p2p_dma=on,num_pci_segments={}",
                self.devices.len() + 1
            );
            command
                .args(["--cpus", cpus.as_str()])
                .args(["--platform", platform.as_str()])
                .args(["--memory", "size=0"])
                .args(["--memory-zone", "id=mem0,size=4G"])
                .args(["--numa", "guest_numa_id=0,cpus=[0-3],memory_zones=mem0"]);
            for (index, device) in self.devices.iter().enumerate() {
                for node in 0..GRACE_GPU_NUMA_NODES {
                    let id = 1 + index * GRACE_GPU_NUMA_NODES + node;
                    command.args([
                        "--numa",
                        &format!("guest_numa_id={id},device_id=gpu{index}"),
                    ]);
                }
                // Behind the emulated SMMUv3, since CUDA needs ATS and PASID,
                // and at device 0 of its own segment, where the driver looks
                // for the GPU's memory.
                command.args([
                    "--device",
                    &format!(
                        "id=gpu{index},path={device},iommu=smmuv3,pci_segment={},\
                         pci_device_id=0",
                        index + 1
                    ),
                ]);
            }
            command.default_disks().default_net().capture_output();
            command
        }
    }

    // nvidia-smi lists every GPU passed through, with no field error. The
    // driver probes a Grace GPU for a while after boot, so retry for two
    // minutes.
    fn check_gpus(guest: &Guest, expected: usize) {
        let start = Instant::now();
        let count = |list: &str| list.lines().filter(|line| line.starts_with("GPU ")).count();
        let (list, exit_code) = loop {
            // A connection that fails this early is retried like the rest.
            let (list, exit_code) = guest
                .ssh_command_exit_code("nvidia-smi -L")
                .unwrap_or_else(|e| (format!("ssh failed: {e}"), -1));
            if (exit_code == 0 && count(&list) == expected)
                || start.elapsed() > Duration::from_secs(120)
            {
                break (list, exit_code);
            }
            thread::sleep(Duration::from_secs(2));
        };
        let errors = nvidia_smi_error_lines(&list);
        assert!(
            exit_code == 0 && errors.is_empty(),
            "nvidia-smi -L: exit code {exit_code}:\n{list}"
        );
        let found = count(&list);
        assert_eq!(
            found,
            expected,
            "nvidia-smi lists {found} GPUs, {expected} passed through, after {} s:\n{list}",
            start.elapsed().as_secs()
        );
        println!("after {} s:\n{list}", start.elapsed().as_secs());
    }

    // Every nvidia-smi view: none may fail or read "Unknown Error".
    #[test]
    #[ignore = "experimental: needs a Grace GPU bound to nvgrace_gpu_vfio_pci and the NVIDIA image in ~/workloads; run with --ignored; #9139"]
    fn test_nvidia_card_nvidia_smi_full() {
        let Some(target) = GpuTarget::discover() else {
            return;
        };
        let guest = Guest::new(Box::new(UbuntuDiskConfig::new(
            NVIDIA_IMAGE_NAME.to_string(),
        )));
        let mut child = target.command(&guest).spawn().unwrap();

        let r = panic::catch_unwind(|| {
            guest.wait_vm_boot().unwrap();
            check_gpus(&guest, target.devices.len());

            let mut problems = guest.nvidia_smi_problems(NVIDIA_SMI_VIEWS);
            problems.extend(guest.nvidia_smi_field_errors(NVIDIA_SMI_HARDWARE_VIEWS));
            assert!(problems.is_empty(), "{}", problems.join("\n"));

            let sweep = guest.nvidia_smi_sweep().unwrap();
            let properties = sweep
                .properties
                .iter()
                .map(|p| p.property.as_str())
                .collect::<BTreeSet<_>>();
            // Queried all properties --help-query-gpu listed, and the list was
            // not empty.
            assert!(
                sweep.listed > 0,
                "nvidia-smi --help-query-gpu listed no property"
            );
            assert_eq!(
                properties.len(),
                sweep.listed,
                "{} of {} listed properties swept",
                properties.len(),
                sweep.listed
            );

            let errors = sweep
                .properties
                .iter()
                .filter(|p| p.class.is_error())
                .collect::<Vec<_>>();
            assert!(errors.is_empty(), "{errors:#?}");
            // Every GPU answered every property.
            let gpus = target.devices.len();
            let short = properties
                .iter()
                .filter(|property| {
                    sweep
                        .properties
                        .iter()
                        .filter(|p| p.property == **property && p.gpu != "-")
                        .count()
                        != gpus
                })
                .collect::<Vec<_>>();
            assert!(
                short.is_empty(),
                "properties not answered for all {gpus} GPUs: {short:?}"
            );

            let not_available = sweep
                .properties
                .iter()
                .filter(|p| !matches!(p.class, NvidiaSmiValue::Value))
                .map(|p| format!("{}={}", p.property, p.value))
                .collect::<BTreeSet<_>>();
            println!(
                "nvidia-smi: {} properties, {} values, without a value: {not_available:?}",
                properties.len(),
                sweep.properties.len()
            );
        });

        let _ = child.kill();
        let output = child.wait_with_output().unwrap();

        handle_child_output(r, &output);
    }

    // "dcgmi diag" selectors with the tests that must run and pass: level 2
    // (software, memory, pcie) and context_create, which is in no level.
    const DCGM_DIAG_SELECTORS: &[(&str, &[&str])] = &[
        ("2", DCGM_DIAG_CORE_TESTS),
        ("context_create", &["context_create"]),
    ];
    // A "dcgmi diag" level 2 run takes a few minutes; one still running after
    // this has hung (seen in nvvs's memory plugin on GH200).
    const DCGM_DIAG_TIMEOUT_SECS: u32 = 1800;
    // The level-2 tests that must pass: the ones that exercise what the VMM
    // presents (the PCIe link, the memory BAR, the driver).
    const DCGM_DIAG_CORE_TESTS: &[&str] = &["software", "memory", "pcie"];

    // Collects the failures of a sequence of guest checks, so that one run
    // reports all of them rather than stopping at the first.
    struct GuestSteps<'a> {
        guest: &'a Guest,
        // The GPUs passed through, which every check must see.
        gpus: usize,
        failures: Vec<String>,
    }

    impl GuestSteps<'_> {
        // Runs `command`, which must exit 0, and returns its output.
        fn run(&mut self, command: &str) -> String {
            match self.guest.ssh_command_exit_code(command) {
                Ok((output, 0)) => output,
                Ok((output, exit_code)) => {
                    self.fail(format!("{command}: exit code {exit_code}:\n{output}"));
                    output
                }
                Err(e) => {
                    self.fail(format!("{command}: {e}"));
                    String::new()
                }
            }
        }

        fn fail(&mut self, failure: String) {
            eprintln!("FAILED: {failure}");
            self.failures.push(failure);
        }
    }

    // Persistence mode onlines a Grace GPU's memory, which CUDA needs; every
    // GPU's numa_status must read "online" within a minute.
    fn online_gpu_memory(steps: &mut GuestSteps) {
        steps.run("sudo nvidia-smi -pm 1");
        let numa_status = steps.run(
            "for i in $(seq 60); do s=$(cat /proc/driver/nvidia/gpus/*/numa_status 2>/dev/null); \
             printf '%s\\n' \"$s\" | grep -qE 'Status: +(offline|[a-z_]+_in_progress)$' || break; \
             sleep 1; done; echo \"$s\"",
        );
        println!("GPU memory NUMA status: {}", numa_status.trim());
        let online = numa_status
            .lines()
            .filter_map(|line| line.strip_prefix("Status:"))
            .filter(|status| status.trim() == "online")
            .count();
        if online != steps.gpus {
            steps.fail(format!(
                "{online} of {} GPUs have their memory online after 60 s: {}",
                steps.gpus,
                numa_status.trim()
            ));
        }
    }

    // Runs "dcgmi diag -r <selector>": no test may fail, and the `required`
    // ones must run and pass on every GPU.
    fn run_dcgm_diag(steps: &mut GuestSteps, selector: &str, required: &[&str]) {
        match steps
            .guest
            .dcgm_diag(&format!("-r {selector}"), DCGM_DIAG_TIMEOUT_SECS)
        {
            Ok(tests) if tests.is_empty() => {
                steps.fail(format!("dcgmi diag -r {selector} ran no test"));
            }
            Ok(tests) => {
                for test in &tests {
                    println!(
                        "dcgmi diag -r {selector}: {} / {}: {}",
                        test.category, test.name, test.status
                    );
                    if test.failed() {
                        steps.fail(format!("dcgmi diag -r {selector}: {test:?}"));
                    }
                }
                for name in required {
                    match tests.iter().find(|test| test.name == *name) {
                        Some(test) if test.passed() || test.failed() => {}
                        Some(test) => steps.fail(format!(
                            "dcgmi diag -r {selector}: {name} must pass on every GPU, \
                             reported {} {:?}",
                            test.status, test.results
                        )),
                        None => {
                            steps.fail(format!("dcgmi diag -r {selector}: {name} did not run"));
                        }
                    }
                }
            }
            Err(e) => steps.fail(format!("dcgmi diag -r {selector}: {e}")),
        }
    }

    // The DCGM checks: every "dcgmi diag" selector runs its tests, none may
    // report Fail and the required ones must pass.
    fn dcgm_test_steps(steps: &mut GuestSteps) {
        for (selector, required) in DCGM_DIAG_SELECTORS {
            run_dcgm_diag(steps, selector, required);
        }
    }

    // Boots the guest, starts DCGM's host engine, which must see every GPU,
    // onlines the GPU memory and runs the DCGM steps.
    fn dcgm_guest_test() {
        let Some(target) = GpuTarget::discover() else {
            return;
        };
        let guest = Guest::new(Box::new(UbuntuDiskConfig::new(
            NVIDIA_IMAGE_NAME.to_string(),
        )));
        let mut child = target.command(&guest).spawn().unwrap();

        let r = panic::catch_unwind(|| {
            guest.wait_vm_boot().unwrap();
            check_gpus(&guest, target.devices.len());

            // The image is built with DCGM: a guest without it is broken.
            assert!(
                guest.dcgm_installed().unwrap(),
                "dcgmi not found in the guest image"
            );
            guest.start_dcgm(target.devices.len()).unwrap();

            let mut steps = GuestSteps {
                guest: &guest,
                gpus: target.devices.len(),
                failures: Vec::new(),
            };
            online_gpu_memory(&mut steps);
            dcgm_test_steps(&mut steps);

            assert!(
                steps.failures.is_empty(),
                "{} DCGM checks failed:\n{}",
                steps.failures.len(),
                steps.failures.join("\n")
            );
        });

        let _ = child.kill();
        let output = child.wait_with_output().unwrap();

        handle_child_output(r, &output);
    }

    // Every DCGM check of dcgm_test_steps() passes.
    #[test]
    #[ignore = "experimental: needs a Grace GPU bound to nvgrace_gpu_vfio_pci and the NVIDIA image with DCGM in ~/workloads; run with --ignored; #9139"]
    fn test_nvidia_card_dcgm() {
        dcgm_guest_test();
    }
}
