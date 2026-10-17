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
        });

        let _ = child.kill();
        let output = child.wait_with_output().unwrap();

        handle_child_output(r, &output);
    }
}
