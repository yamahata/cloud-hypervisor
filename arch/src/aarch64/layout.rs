// Copyright 2020 Arm Limited (or its affiliates). All rights reserved.
// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Memory layout of AArch64 guest:
//
// Physical  +---------------------------------------------------------------+
// address   |                                                               |
// end       |                                                               |
//           ~                   ~                       ~                   ~
//           |                                                               |
//           |                      Highmem PCI MMIO space                   |
//           |                                                               |
// RAM end   +---------------------------------------------------------------+
// (dynamic, |                                                               |
// including |                                                               |
// hotplug   ~                   ~                       ~                   ~
// memory)   |                                                               |
//           |                            DRAM                               |
//           |                                                               |
//           |                                                               |
// 4GB       +---------------------------------------------------------------+
//           |                      32-bit devices hole                      |
// 4GB-64M   +---------------------------------------------------------------+
//           |                                                               |
//           |                                                               |
//           |                            DRAM                               |
//           |                                                               |
//           |                                                               |
// 1GB       +---------------------------------------------------------------+
//           |                                                               |
//           |                        PCI MMCONFIG space                     |
//           |                                                               |
// 768 M     +---------------------------------------------------------------+
//           |                                                               |
//           |                                                               |
//           |                           PCI MMIO space                      |
//           |                                                               |
// 256 M     +---------------------------------------------------------------|
//           |                                                               |
//           |                        Legacy devices space                   |
//           |                                                               |
// 144 M     +---------------------------------------------------------------|
//           |                GICv3 Distributor (0x08ff_0000)                |
// 143.9M    +---------------------------------------------------------------|
//           |   Legacy GICv3 redistributors + ITS, 128 KiB/vCPU, growing    |
//           |   DOWN from the distributor (guests with no SMMUv3). No       |
//           |   fixed floor: reaches ~112 M at MAX_SUPPORTED_CPUS = 255,    |
//           |   through the SMMUv3 RMR window and Reserved bands below      |
//           |   (harmless: such a guest has no RMR window of its own).      |
//           +---------------------------------------------------------------|
//           |                 SMMUv3 RMR MSI identity window                |
// 128 M     +---------------------------------------------------------------|
//           |                          Reserved                             |
//  48 M     +---------------------------------------------------------------|
//           |   GICv3 redistributors + ITS for SMMUv3 guests, growing UP.   |
//           |   ALTERNATIVE to the legacy band above: a guest uses ONE      |
//           |   placement or the other, never both. (48 M is the array      |
//           |   end at MAX_SUPPORTED_CPUS = 255.)                           |
//  16 M     +---------------------------------------------------------------|
//           |                          Reserved                             |
//  4  M     +---------------------------------------------------------------+
//           |                          UEFI flash                           |
// 0GB       +---------------------------------------------------------------+
//
//

use vm_memory::GuestAddress;

/// 0x0 ~ 0x40_0000 (4 MiB) is reserved to UEFI
/// UEFI binary size is required less than 3 MiB, reserving 4 MiB is enough.
pub const UEFI_START: GuestAddress = GuestAddress(0);
pub const UEFI_SIZE: u64 = 0x040_0000;

/// Below this address will reside the GIC, above this address will reside the MMIO devices.
const MAPPED_IO_START: GuestAddress = GuestAddress(0x0900_0000);

/// See kernel file arch/arm64/include/uapi/asm/kvm.h for the GIC related definitions.
/// 0x08ff_0000 ~ 0x0900_0000 is reserved for GICv3 Distributor.
///
/// The distributor never moves. It is the only GIC address that has
/// ever been constant - the redistributor and ITS bases have always
/// been functions of the vCPU count - so it is the only one firmware
/// could plausibly have hardcoded rather than read from the FDT.
pub const GIC_V3_DIST_SIZE: u64 = 0x01_0000;
pub const GIC_V3_DIST_START: GuestAddress = GuestAddress(MAPPED_IO_START.0 - GIC_V3_DIST_SIZE);

/// The size defined here is for each vCPU; the total is
/// 'number_of_vcpu * GIC_V3_REDIST_SIZE'.
pub const GIC_V3_REDIST_SIZE: u64 = 0x02_0000;
/// Size of the GICv3 ITS frame.
pub const GIC_V3_ITS_SIZE: u64 = 0x02_0000;

/// Which GICv3 redistributor/ITS placement a guest's memory map uses.
///
/// The redistributor array is `vcpu_count * GIC_V3_REDIST_SIZE` long
/// and has to go somewhere that does not collide with the SMMUv3 RMR
/// MSI identity window at `RMR_MSI_WINDOW_BASE`.
///
/// - `Legacy` is the placement every cloud-hypervisor has used: the
///   array grows *downward* from the distributor, with the ITS directly
///   below it. It reaches the RMR window at 119 vCPUs.
/// - `BelowRmrWindow` puts the ITS and the array at fixed bases below
///   the window, with the array growing *upward* toward it, leaving
///   room for `GIC_V3_MAX_VCPUS` vCPUs.
///
/// A guest without an SMMUv3 has no RMR window in its memory map, so it
/// keeps `Legacy` - which is what lets it be live-migrated from an
/// older cloud-hypervisor, since a guest carries its GIC addresses in
/// page tables and MSI-X entries that no migration can rewrite. A guest
/// *with* an SMMUv3 cannot be restored or migrated: the emulated
/// SMMUv3's nested state is not snapshotted (see `docs/iommu.md`), so
/// relocating its GIC costs nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GicV3Placement {
    Legacy,
    BelowRmrWindow,
}

/// Fixed base of the GICv3 ITS under `GicV3Placement::BelowRmrWindow`.
/// 16 MiB leaves a cushion above the UEFI flash, whose own reservation
/// is already sized for growth.
pub const GIC_V3_ITS_START: GuestAddress = GuestAddress(0x0100_0000);
/// Fixed base of the redistributor array under
/// `GicV3Placement::BelowRmrWindow`: directly above the ITS.
pub const GIC_V3_REDIST_START: GuestAddress = GuestAddress(GIC_V3_ITS_START.0 + GIC_V3_ITS_SIZE);
/// The largest vCPU count whose redistributor array still fits below
/// `RMR_MSI_WINDOW_BASE` under `GicV3Placement::BelowRmrWindow`.
/// `vmm::config` asserts at compile time that `MAX_SUPPORTED_CPUS` does
/// not exceed it.
pub const GIC_V3_MAX_VCPUS: u64 =
    (RMR_MSI_WINDOW_BASE - GIC_V3_REDIST_START.0) / GIC_V3_REDIST_SIZE;

/// Base address of the redistributor array for `vcpu_count` vCPUs.
/// Kept here, next to the constants it derives from, so that anything
/// needing to reason about these boundaries shares one arithmetic
/// instead of re-deriving (and drifting from) it.
#[inline]
pub const fn gic_v3_redist_start(placement: GicV3Placement, vcpu_count: u64) -> u64 {
    match placement {
        GicV3Placement::Legacy => GIC_V3_DIST_START.0 - vcpu_count * GIC_V3_REDIST_SIZE,
        GicV3Placement::BelowRmrWindow => GIC_V3_REDIST_START.0,
    }
}

/// Base address of the GICv3 ITS for `vcpu_count` vCPUs. Mirrors
/// `Gic::create_default_config`'s placement exactly.
#[inline]
pub const fn gic_v3_its_start(placement: GicV3Placement, vcpu_count: u64) -> u64 {
    match placement {
        GicV3Placement::Legacy => {
            gic_v3_redist_start(GicV3Placement::Legacy, vcpu_count) - GIC_V3_ITS_SIZE
        }
        GicV3Placement::BelowRmrWindow => GIC_V3_ITS_START.0,
    }
}

/// The SMMUv3 RMR MSI identity window: a hole in the guest memory map
/// where the host kernel composes a bound device's MSI doorbell
/// address (see `docs/smmuv3.md`'s "Flow" section) and which every
/// SMMUv3-bound device's IORT RMR node identity-maps.
///
/// This sits above the ITS/redistributor region and below the GIC
/// distributor (`GIC_V3_DIST_START`) for every vCPU count: a guest that
/// has this window uses `GicV3Placement::BelowRmrWindow`, whose
/// redistributor array is bounded by `GIC_V3_MAX_VCPUS` at compile
/// time, so it can never grow up into this window.
pub const RMR_MSI_WINDOW_BASE: u64 = 0x0800_0000;
pub const RMR_MSI_WINDOW_SIZE: u64 = 0x0010_0000;

/// Space 0x0900_0000 ~ 0x0905_0000 is reserved for legacy devices.
pub const LEGACY_SERIAL_MAPPED_IO_START: GuestAddress = MAPPED_IO_START;
pub const LEGACY_RTC_MAPPED_IO_START: GuestAddress = GuestAddress(0x0901_0000);
pub const LEGACY_GPIO_MAPPED_IO_START: GuestAddress = GuestAddress(0x0902_0000);

/// Space 0x0905_0000 ~ 0x0906_0000 is reserved for pcie io address
pub const MEM_PCI_IO_START: GuestAddress = GuestAddress(0x0905_0000);
pub const MEM_PCI_IO_SIZE: u64 = 0x10000;

/// Starting from 0x1000_0000 (256MiB) to 0x3000_0000 (768MiB) is used for PCIE MMIO
pub const MEM_32BIT_DEVICES_START: GuestAddress = GuestAddress(0x1000_0000);
pub const MEM_32BIT_DEVICES_SIZE: u64 = 0x2000_0000;

// Compile-time sanity checks for the memory layout.
const _: () = assert!(
    UEFI_START.0 + UEFI_SIZE <= GIC_V3_ITS_START.0,
    "the relocated GIC ITS/redistributor region must not overlap the UEFI flash"
);
const _: () = assert!(
    GIC_V3_ITS_START.0.is_multiple_of(0x1_0000) && GIC_V3_REDIST_START.0.is_multiple_of(0x1_0000),
    "KVM requires 64 KiB-aligned GIC ITS and redistributor bases"
);
const _: () = assert!(
    GIC_V3_REDIST_START.0 < RMR_MSI_WINDOW_BASE,
    "the relocated redistributor array must start below the RMR MSI window"
);
const _: () = assert!(
    RMR_MSI_WINDOW_BASE + RMR_MSI_WINDOW_SIZE <= GIC_V3_DIST_START.0,
    "the RMR MSI window must stay below the GIC distributor"
);
const _: () = assert!(
    MEM_32BIT_DEVICES_START.0 == 0x1000_0000 && MEM_32BIT_DEVICES_SIZE == 0x2000_0000,
    "the 32-bit PCI MMIO window is part of the migration contract: a VM restored or \
     migrated from another build keeps its BARs, which must stay inside the window"
);
const _: () = assert!(
    MEM_32BIT_DEVICES_START.0 + MEM_32BIT_DEVICES_SIZE <= PCI_MMCONFIG_START.0,
    "PCI MMIO space must not overlap with PCI MMCONFIG space"
);
/// PCI MMCONFIG space (start: after the device space at 1 GiB, length: 256MiB)
pub const PCI_MMCONFIG_START: GuestAddress = GuestAddress(0x3000_0000);
pub const PCI_MMCONFIG_SIZE: u64 = 256 << 20;
// One bus with potentially 256 devices (32 slots x 8 functions).
pub const PCI_MMIO_CONFIG_SIZE_PER_SEGMENT: u64 = 4096 * 256;

/// Start of RAM.
pub const RAM_START: GuestAddress = GuestAddress(0x4000_0000);

/// 32-bit reserved area: 64MiB before 4GiB
pub const MEM_32BIT_RESERVED_START: GuestAddress = GuestAddress(0xfc00_0000);
pub const MEM_32BIT_RESERVED_SIZE: u64 = 0x0400_0000;

/// TPM Address Range
/// This Address range is specific to CRB Interface
pub const TPM_START: GuestAddress = GuestAddress(0xfed4_0000);
pub const TPM_SIZE: u64 = 0x1000;

/// Start of 64-bit RAM.
pub const RAM_64BIT_START: GuestAddress = GuestAddress(0x1_0000_0000);

/// Kernel command line maximum size.
/// As per `arch/arm64/include/uapi/asm/setup.h`.
pub const CMDLINE_MAX_SIZE: usize = 2048;

/// FDT is at the beginning of RAM.
pub const FDT_START: GuestAddress = RAM_START;
/// Maximum size of the device tree blob as specified in [the kernel
/// documentation](https://www.kernel.org/doc/Documentation/arm64/booting.txt).
pub const FDT_MAX_SIZE: u64 = 0x20_0000;

/// Put ACPI table above dtb
pub const ACPI_START: GuestAddress = GuestAddress(RAM_START.0 + FDT_MAX_SIZE);
const ACPI_SMBIOS_MAX_SIZE: u64 = 0x20_0000;
pub const ACPI_MAX_SIZE: u64 = ACPI_SMBIOS_MAX_SIZE - SMBIOS_MAX_SIZE;
pub const RSDP_POINTER: GuestAddress = ACPI_START;

/// Put SMBIOS table above ACPI table and below the kernel
pub const SMBIOS_START: GuestAddress = GuestAddress(ACPI_START.0 + ACPI_MAX_SIZE);
pub const SMBIOS_MAX_SIZE: u64 = 0x1_0000;

/// Kernel start after the above
pub const KERNEL_START: GuestAddress = GuestAddress(SMBIOS_START.0 + SMBIOS_MAX_SIZE);

/// Pci high memory base
pub const PCI_HIGH_BASE: GuestAddress = GuestAddress(0x2_0000_0000);

// As per virt/kvm/arm/vgic/vgic-kvm-device.c we need
// the number of interrupts our GIC will support to be:
// * bigger than 32
// * less than 1023 and
// * a multiple of 32.
// We are setting up our interrupt controller to support a maximum of 256 interrupts.
/// First usable interrupt on aarch64
pub const IRQ_BASE: u32 = 32;

/// Number of supported interrupts
pub const IRQ_NUM: u32 = 256;

/// Base SPI interrupt number for the GICv2M MSI frame
pub const GICV2M_SPI_BASE: u32 = 128;

/// Total number of SPIs for the GICv2M MSI frame
pub const GICV2M_SPI_NUM: u32 = 64;

/// GICv2M compatible string
pub const GIC_V2M_COMPATIBLE: &str = "arm,gic-v2m-frame";
