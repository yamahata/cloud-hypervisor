// Copyright © 2026 Cloud Hypervisor Authors
// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA Tegra241 CMDQ-Virtualization (CMDQV) device model.
//
// CMDQV is a per-SMMUv3-instance NVIDIA extension that lets the guest
// drive a hardware command queue (VCMDQ) directly - TLBI/ATC_INV
// commands the guest enqueues execute without a VMM trap on the
// doorbell, rather than being decoded and forwarded by the emulated
// SMMUv3 command queue. This is the analogue of QEMU's
// hw/arm/tegra241-cmdqv.c; the register offsets, reset values and the
// `-0x20000` VI_VCMDQ alias fold below MUST match
// qemu-ref/tegra241-cmdqv.h byte-for-byte.
//
// This module is the register-cache emulation core, the VCMDQ page0
// mmap mirror, and hardware queue allocation on VCMDQ BASE writes.

use std::iter::repeat_with;
use std::ptr;
use std::sync::{Arc, Barrier};

use anyhow::anyhow;
use iommufd_ioctls::{IommufdHwQueue, IommufdVIommu};
use log::{debug, info, trace, warn};
use pci::mmap::MmapRegion;
use vm_device::BusDevice;
use vm_memory::bitmap::AtomicBitmap;
use vm_memory::{
    GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryAtomic, GuestMemoryMmap, Permissions,
};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};

/// Number of VCMDQs (and CMDQ_ALLOC_MAP entries) the register space
/// decodes, architecturally: `CMDQ_ALLOC_MAP_i.LVCMDQ` is 7 bits and
/// `VCMDQi` offsets are only defined up to index 127
/// (tegra241-cmdqv.h). Real Tegra241 hardware only ever implements 2
/// (PARAM.CMDQV_NUM_CMDQ_LOG2 == 1, asserted against the host hw-info,
/// design §6), but the decoder honours the full architectural range
/// like QEMU's `vcmdq[128]`.
const NUM_VCMDQ: usize = 128;

/// MMIO register offsets, mirroring `tegra241-cmdqv.h` byte-for-byte.
mod offset {
    pub(super) const CONFIG: u64 = 0x0000;
    pub(super) const PARAM: u64 = 0x0004;
    pub(super) const STATUS: u64 = 0x0008;

    pub(super) const VI_ERR_MAP: u64 = 0x0014;
    pub(super) const VI_ERR_MAP_1: u64 = 0x0018;

    pub(super) const VI_INT_MASK: u64 = 0x001c;
    pub(super) const VI_INT_MASK_1: u64 = 0x0020;

    pub(super) const CMDQ_ERR_MAP: u64 = 0x0024;
    pub(super) const CMDQ_ERR_MAP_3: u64 = 0x0030;

    pub(super) const CMDQ_ALLOC_MAP_0: u64 = 0x0200;
    pub(super) const CMDQ_ALLOC_MAP_127: u64 = 0x0200 + 127 * 4; // 0x03fc

    pub(super) const VINTF0_CONFIG: u64 = 0x1000;
    pub(super) const VINTF0_STATUS: u64 = 0x1004;
    pub(super) const VINTF0_LVCMDQ_ERR_MAP_0: u64 = 0x10c0;
    pub(super) const VINTF0_LVCMDQ_ERR_MAP_3: u64 = 0x10c0 + 3 * 4; // 0x10cc

    /// VCMDQ page0 (per-index CONS_INDX/PROD_INDX/CONFIG/STATUS/
    /// GERROR/GERRORN): `0x10000 + i * 0x80 + reg`.
    pub(super) const VCMDQ_PAGE0_BASE: u64 = 0x10000;
    /// VCMDQ page1 (per-index BASE_L/BASE_H/CONS_INDX_BASE_DRAM_L/_H):
    /// `0x20000 + i * 0x80 + reg`.
    pub(super) const VCMDQ_PAGE1_BASE: u64 = 0x20000;
    /// VI_VCMDQ page0: mirrors VCMDQ page0 at a `+0x20000` alias.
    pub(super) const VI_VCMDQ_PAGE0_BASE: u64 = 0x30000;
    // VI_VCMDQ page1 (0x40000, mirrors VCMDQ page1 at a `+0x20000`
    // alias) needs no separate constant: it is exactly `SPAN_LIMIT -
    // VCMDQ_PAGE1_BASE + VI_VCMDQ_PAGE0_BASE`, i.e. covered by the
    // `VI_VCMDQ_PAGE0_BASE..SPAN_LIMIT` fold range below.

    /// Per-index stride within a VCMDQ/VI_VCMDQ page.
    pub(super) const VCMDQ_STRIDE: u64 = 0x80;
    /// The VI_VCMDQ alias pages fold onto the native VCMDQ pages they
    /// mirror by subtracting this amount.
    pub(super) const FOLD: u64 = 0x20000;
    /// Byte past the last offset the device decodes (`0x50000`); QEMU
    /// logs and ignores any access at or beyond this.
    pub(super) const SPAN_LIMIT: u64 = 0x50000;

    /* Register offsets within a VCMDQ/VI_VCMDQ page0 stride. */
    pub(super) const VCMDQ_CONS_INDX: u64 = 0x00;
    pub(super) const VCMDQ_PROD_INDX: u64 = 0x04;
    pub(super) const VCMDQ_CONFIG: u64 = 0x08;
    pub(super) const VCMDQ_STATUS: u64 = 0x0c;
    pub(super) const VCMDQ_GERROR: u64 = 0x10;
    pub(super) const VCMDQ_GERRORN: u64 = 0x14;

    /* Register offsets within a VCMDQ/VI_VCMDQ page1 stride. */
    pub(super) const VCMDQ_BASE_L: u64 = 0x00;
    pub(super) const VCMDQ_BASE_H: u64 = 0x04;
    pub(super) const VCMDQ_CONS_INDX_BASE_DRAM_L: u64 = 0x08;
    pub(super) const VCMDQ_CONS_INDX_BASE_DRAM_H: u64 = 0x0c;
}

/// CONFIG reset value (`V_CONFIG_RESET` in `tegra241-cmdqv.h`):
/// CMDQV_EN=1 plus the vendor's fixed batching defaults.
const CONFIG_RESET: u32 = 0x0002_0403;
/// PARAM reset value (`V_PARAM_RESET`): CMDQV_VER=1,
/// CMDQV_NUM_CMDQ_LOG2=1 (2 VCMDQs), CMDQV_NUM_VM_LOG2=0 (1 VINTF),
/// CMDQV_NUM_SID_PER_VM_LOG2=4 (16 SIDs) - the values design §6
/// validates against the host's `IOMMU_HW_INFO_TYPE_TEGRA241_CMDQV`.
const PARAM_RESET: u32 = 0x0000_4011;
/// STATUS reset value: CMDQV_ENABLED mirrors CONFIG.CMDQV_EN, which is
/// set in `CONFIG_RESET`.
const STATUS_RESET: u32 = 0x1;

/// CONFIG.CMDQV_EN (bit 0).
const CONFIG_CMDQV_EN_MASK: u32 = 1 << 0;
/// STATUS.CMDQV_ENABLED (bit 0).
const STATUS_CMDQV_ENABLED_MASK: u32 = 1 << 0;
/// VINTFi_CONFIG.ENABLE (bit 0).
const VINTF_ENABLE_MASK: u32 = 1 << 0;
/// VINTFi_CONFIG.HYP_OWN (bit 17): the guest kernel is not the
/// hypervisor, so this bit is always stripped from a guest write
/// (tegra241-cmdqv.c:411-413) - the VMM, not the guest, owns the
/// VINTF.
const VINTF_HYP_OWN_MASK: u32 = 1 << 17;
/// VINTFi_STATUS.ENABLE_OK (bit 0).
const VINTF_STATUS_ENABLE_OK_MASK: u32 = 1 << 0;
/// VCMDQi_CONFIG.CMDQ_EN (bit 0).
const VCMDQ_CONFIG_CMDQ_EN_MASK: u32 = 1 << 0;
/// VCMDQi_STATUS.CMDQ_EN_OK (bit 0).
const VCMDQ_STATUS_CMDQ_EN_OK_MASK: u32 = 1 << 0;

/// A decoded register access, after folding the VI_VCMDQ alias pages
/// onto the native VCMDQ pages they mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reg {
    Config,
    Param,
    Status,
    ViErrMap(usize),
    ViIntMask(usize),
    CmdqErrMap(usize),
    CmdqAllocMap(usize),
    Vintf0Config,
    Vintf0Status,
    Vintf0LvcmdqErrMap(usize),
    /// CONS_INDX/PROD_INDX/CONFIG/STATUS/GERROR/GERRORN for VCMDQ
    /// `index`, register byte offset `reg` within the `0x80` stride.
    VcmdqPage0 {
        index: usize,
        reg: u64,
    },
    /// BASE_L/BASE_H/CONS_INDX_BASE_DRAM_L/_H for VCMDQ `index`,
    /// register byte offset `reg` within the `0x80` stride.
    VcmdqPage1 {
        index: usize,
        reg: u64,
    },
}

/// Decode an offset within the CMDQV's `0x50000`-byte MMIO span.
/// Returns `None` for offsets the device does not implement (either
/// out of range, or a gap between defined registers) - the caller
/// logs and ignores these, matching QEMU's `LOG_UNIMP` fallback.
fn decode(raw_offset: u64) -> Option<Reg> {
    // Fold the VI_VCMDQ alias pages (0x30000..0x4ffff) onto the native
    // VCMDQ pages (0x10000..0x2ffff) they mirror: the guest driver
    // reaches VCMDQ state through either alias interchangeably
    // (tegra241-cmdqv.c:370-398/628-660).
    let offset = if (offset::VI_VCMDQ_PAGE0_BASE..offset::SPAN_LIMIT).contains(&raw_offset) {
        raw_offset - offset::FOLD
    } else {
        raw_offset
    };

    match offset {
        offset::CONFIG => Some(Reg::Config),
        offset::PARAM => Some(Reg::Param),
        offset::STATUS => Some(Reg::Status),
        offset::VI_ERR_MAP..=offset::VI_ERR_MAP_1 => {
            Some(Reg::ViErrMap(((offset - offset::VI_ERR_MAP) / 4) as usize))
        }
        offset::VI_INT_MASK..=offset::VI_INT_MASK_1 => Some(Reg::ViIntMask(
            ((offset - offset::VI_INT_MASK) / 4) as usize,
        )),
        offset::CMDQ_ERR_MAP..=offset::CMDQ_ERR_MAP_3 => Some(Reg::CmdqErrMap(
            ((offset - offset::CMDQ_ERR_MAP) / 4) as usize,
        )),
        offset::CMDQ_ALLOC_MAP_0..=offset::CMDQ_ALLOC_MAP_127 => Some(Reg::CmdqAllocMap(
            ((offset - offset::CMDQ_ALLOC_MAP_0) / 4) as usize,
        )),
        offset::VINTF0_CONFIG => Some(Reg::Vintf0Config),
        offset::VINTF0_STATUS => Some(Reg::Vintf0Status),
        offset::VINTF0_LVCMDQ_ERR_MAP_0..=offset::VINTF0_LVCMDQ_ERR_MAP_3 => Some(
            Reg::Vintf0LvcmdqErrMap(((offset - offset::VINTF0_LVCMDQ_ERR_MAP_0) / 4) as usize),
        ),
        _ if (offset::VCMDQ_PAGE0_BASE..offset::VCMDQ_PAGE1_BASE).contains(&offset) => {
            let rel = offset - offset::VCMDQ_PAGE0_BASE;
            let index = (rel / offset::VCMDQ_STRIDE) as usize;
            (index < NUM_VCMDQ).then_some(Reg::VcmdqPage0 {
                index,
                reg: rel % offset::VCMDQ_STRIDE,
            })
        }
        _ if (offset::VCMDQ_PAGE1_BASE..offset::VI_VCMDQ_PAGE0_BASE).contains(&offset) => {
            let rel = offset - offset::VCMDQ_PAGE1_BASE;
            let index = (rel / offset::VCMDQ_STRIDE) as usize;
            (index < NUM_VCMDQ).then_some(Reg::VcmdqPage1 {
                index,
                reg: rel % offset::VCMDQ_STRIDE,
            })
        }
        _ => None,
    }
}

/// Extract a register value from a guest access, truncating to the
/// low 32 bits for an 8-byte access. This matches the trapped
/// registers' C-level semantics in `tegra241-cmdqv.c`: `value` is
/// assigned directly to a `uint32_t` field regardless of the access
/// `size` that produced it, so a stray 8-byte access to a 32-bit
/// register silently drops the high bits rather than spilling into
/// the next register (unlike `SMMUv3::write`'s generic two-way split
/// - see `devices/src/vsmmuv3/device.rs`).
fn value32(data: &[u8]) -> u32 {
    let mut bytes = [0u8; 8];
    bytes[..data.len()].copy_from_slice(data);
    u32::from_le_bytes(bytes[0..4].try_into().unwrap())
}

/// Extract the full 64-bit value from an 8-byte guest access.
fn value64(data: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..data.len()].copy_from_slice(data);
    u64::from_le_bytes(bytes)
}

/// Whether `[addr, addr + size)` lies entirely within guest RAM.
/// Split out from `setup_vcmdq` so it can be unit-tested without a
/// live vIOMMU (`tegra241_cmdqv_setup_vcmdq()`'s
/// `cpu_physical_memory_is_ram()` guard).
fn addr_in_guest_ram(
    mem: &GuestMemoryAtomic<GuestMemoryMmap<AtomicBitmap>>,
    addr: u64,
    size: u64,
) -> bool {
    let Ok(size) = usize::try_from(size) else {
        return false;
    };
    mem.memory()
        .check_range(GuestAddress(addr), size, Permissions::ReadWrite)
}

/// Whether a VCMDQ's negotiated `size` (`1u64 << (log2size + 4)`, so
/// never zero) fits within `cap`. The kernel requires a VCMDQ's queue
/// memory to be physically contiguous, so a queue can be no larger than
/// the granule guest RAM is backed by: the host page, or the explicit
/// hugepage size when every memory zone uses one. The device manager
/// derives that granule once per instance and uses it both to bound the
/// owning SMMUv3's IDR1.CMDQS (from which the guest driver sizes its
/// VCMDQs) and, through `Tegra241Cmdqv::set_queue_size_cap`, as `cap`
/// here, so the size the guest may pick and the size this device will
/// allocate always agree. Split out from `setup_vcmdq` so it can be
/// unit-tested without a live vIOMMU, mirroring `addr_in_guest_ram`.
fn size_fits_cap(size: u64, cap: u64) -> bool {
    size <= cap
}

/// The host Tegra241 CMDQV VINTF page0 mmap: the same host page the
/// iommufd crate's `IommufdVIommu::data()` coordinates describe, mapped
/// eagerly by the device manager and handed to the device with
/// `Tegra241Cmdqv::set_vintf_page`. Backs the VCMDQ page0 registers
/// (CONS_INDX/PROD_INDX/CONFIG/STATUS/GERROR/GERRORN) at both the
/// `+0x10000` and `+0x30000` guest offsets (design §1.2) - both fold
/// to the same `Reg::VcmdqPage0` decode, so one `VintfPage` serves
/// both.
///
/// Normally a KVM memslot aliases this same host page directly into
/// guest-physical space, so the guest's doorbell writes never reach
/// this device's `BusDevice` impl at all (the whole point of CMDQV).
/// This holder exists for the correctness fallback: if the memslot
/// cannot be created, VCMDQ page0 accesses keep trapping here, and are
/// mirrored into the mmap "manually" so the guest still observes real
/// hardware state - at the cost of the exitless doorbell.
struct VintfPage {
    mapping: MmapRegion,
}

impl VintfPage {
    fn new(mapping: MmapRegion) -> Self {
        VintfPage { mapping }
    }

    /// Read the 32-bit value at `byte_offset` into the mapped page.
    fn read_u32(&self, byte_offset: usize) -> u32 {
        debug_assert!(byte_offset + 4 <= self.mapping.len());
        // SAFETY: `byte_offset + 4 <= self.mapping.len()` (checked
        // above), and `MmapRegion`'s invariants guarantee `addr()`
        // points to `len()` bytes of valid, MAP_SHARED memory for the
        // mapping's lifetime.
        unsafe { ptr::read_volatile(self.mapping.addr().add(byte_offset).cast::<u32>()) }
    }

    /// Write the 32-bit `value` at `byte_offset` into the mapped page.
    fn write_u32(&self, byte_offset: usize, value: u32) {
        debug_assert!(byte_offset + 4 <= self.mapping.len());
        // SAFETY: see `read_u32`.
        unsafe { ptr::write_volatile(self.mapping.addr().add(byte_offset).cast::<u32>(), value) }
    }
}

/// Tegra241 CMDQV device state: the register cache backing the
/// trapped registers, indexed exactly as `tegra241-cmdqv.h` describes.
pub struct Tegra241Cmdqv {
    /// Instance name (e.g., "vsmmuv3_0_cmdqv"), for logging.
    name: String,
    /// Owning SMMUv3 instance index; also this CMDQV's ACPI `_UID`.
    instance_id: u32,
    /// SPI allocated for this instance at creation time (design §5,
    /// Q-G: advertised now, unconditionally, even though nothing
    /// asserts it before the deferred vEVENTQ follow-up).
    irq: u32,
    /// Whether the device manager has finished wiring this instance
    /// up: a vIOMMU is attached, its Tegra241 CMDQV hw-info validated
    /// against the host, and the VINTF page mapped (regardless of
    /// whether the memslot fast path or the trap-and-mirror fallback
    /// ended up backing it). Distinct from the guest-controlled
    /// CONFIG.CMDQV_EN register bit; drives DSDT's conditional
    /// emission (design §5: a CMDQV node is only ever safe to
    /// advertise once this is true - QEMU's unbacked-CMDQV fallback
    /// lies about CMDQ_EN_OK and hangs the guest).
    enabled: bool,
    /// The host VINTF page0 mmap, once mapped (see `VintfPage`).
    /// `None` until `set_vintf_page` is called, and always `None` in
    /// unit tests, which exercise the register-cache-only fallback.
    vintf_page: Option<VintfPage>,
    /// KVM memslot ids created to alias the host VINTF page0 into
    /// guest-physical space (design §3/§7 item 5): zero, one or two
    /// entries, one per alias offset (+0x10000/+0x30000) whose memslot
    /// creation succeeded - each is attempted independently, so either,
    /// both or neither may be present. Bookkeeping only, set once by
    /// `set_memslots`: explicit removal is not required because these
    /// memslots alias process memory the VMM itself owns and are torn
    /// down automatically when the VM's `VmFd` is dropped, not a
    /// device-specific resource this struct's `Drop` needs to unwind.
    memslots: Vec<u32>,

    config: u32,
    param: u32,
    status: u32,
    vi_err_map: [u32; 2],
    vi_int_mask: [u32; 2],
    cmdq_err_map: [u32; 4],
    cmdq_alloc_map: [u32; NUM_VCMDQ],
    vintf_config: u32,
    vintf_status: u32,
    vintf_lvcmdq_err_map: [u32; 4],
    vcmdq_cons_indx: [u32; NUM_VCMDQ],
    vcmdq_prod_indx: [u32; NUM_VCMDQ],
    vcmdq_config: [u32; NUM_VCMDQ],
    vcmdq_status: [u32; NUM_VCMDQ],
    vcmdq_gerror: [u32; NUM_VCMDQ],
    vcmdq_gerrorn: [u32; NUM_VCMDQ],
    vcmdq_base: [u64; NUM_VCMDQ],
    vcmdq_cons_indx_base: [u64; NUM_VCMDQ],

    /// Guest memory, to validate a VCMDQ's negotiated (addr, size)
    /// against guest RAM before allocating a hardware queue over it.
    mem: GuestMemoryAtomic<GuestMemoryMmap<AtomicBitmap>>,
    /// The vIOMMU backing this instance's hardware queues. `None`
    /// until `set_viommu` is called (mirrors `Smmuv3Accel::viommu`);
    /// a VCMDQ BASE write with no vIOMMU attached yet only latches the
    /// register cache (see `setup_vcmdq`).
    viommu: Option<Arc<IommufdVIommu>>,
    /// Allocated hardware queues, one slot per architectural VCMDQ
    /// index. Real hardware only ever populates 2 of these
    /// (PARAM.CMDQV_NUM_CMDQ_LOG2 == 1), but the guest may in
    /// principle program any of the 128 architectural indices.
    vcmdq: Vec<Option<IommufdHwQueue>>,
    /// Largest VCMDQ this device allocates, in bytes (see
    /// `size_fits_cap`). The host page unless the device manager raises
    /// it with `set_queue_size_cap`.
    queue_size_cap: u64,
}

impl Tegra241Cmdqv {
    /// Create a new CMDQV device for the SMMUv3 instance `instance_id`,
    /// with SPI `irq` already allocated (design §5, §7).
    pub fn new(
        name: String,
        instance_id: u32,
        irq: u32,
        mem: GuestMemoryAtomic<GuestMemoryMmap<AtomicBitmap>>,
    ) -> Self {
        Tegra241Cmdqv {
            name,
            instance_id,
            irq,
            enabled: false,
            vintf_page: None,
            memslots: Vec::new(),
            config: CONFIG_RESET,
            param: PARAM_RESET,
            status: STATUS_RESET,
            vi_err_map: [0; 2],
            vi_int_mask: [0; 2],
            cmdq_err_map: [0; 4],
            cmdq_alloc_map: [0; NUM_VCMDQ],
            vintf_config: 0,
            vintf_status: 0,
            vintf_lvcmdq_err_map: [0; 4],
            vcmdq_cons_indx: [0; NUM_VCMDQ],
            vcmdq_prod_indx: [0; NUM_VCMDQ],
            vcmdq_config: [0; NUM_VCMDQ],
            vcmdq_status: [0; NUM_VCMDQ],
            vcmdq_gerror: [0; NUM_VCMDQ],
            vcmdq_gerrorn: [0; NUM_VCMDQ],
            vcmdq_base: [0; NUM_VCMDQ],
            vcmdq_cons_indx_base: [0; NUM_VCMDQ],
            mem,
            viommu: None,
            vcmdq: repeat_with(|| None).take(NUM_VCMDQ).collect(),
            // SAFETY: sysconf(_SC_PAGESIZE) has no side effects.
            queue_size_cap: unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64,
        }
    }

    /// Raise (or lower) the largest VCMDQ this device will allocate to
    /// `bytes`, the granule guest RAM is backed by. Must match the bound
    /// applied to the owning SMMUv3's IDR1.CMDQS; see `size_fits_cap`.
    pub fn set_queue_size_cap(&mut self, bytes: u64) {
        self.queue_size_cap = bytes;
    }

    /// Device name (ID), for logging.
    pub fn id(&self) -> &str {
        &self.name
    }

    /// The owning SMMUv3 instance index (also this device's ACPI
    /// `_UID` - see design §5).
    pub fn instance_id(&self) -> u32 {
        self.instance_id
    }

    /// The SPI allocated for this instance at creation time.
    pub fn irq(&self) -> u32 {
        self.irq
    }

    /// Mark this instance as fully wired up (or not): a vIOMMU
    /// attached, hw-info validated, and the VINTF page mapped. Called
    /// once, by the device manager, after all three of those succeed,
    /// never before and never partially (see the `enabled` field doc
    /// comment).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether this instance is fully wired up - see `set_enabled`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// MMIO span this device decodes (`0x50000`); the guest-visible
    /// stride between instances is wider (`layout::
    /// SMMUV3_CMDQV_INSTANCE_SIZE`) to leave headroom for growth.
    pub fn mmio_size() -> u64 {
        offset::SPAN_LIMIT
    }

    /// Back the VCMDQ page0 registers with the host VINTF page0 mmap.
    /// Idempotent-in-intent but not idempotent-in-fact: called once,
    /// by the device manager, after the memslot(s) aliasing this same
    /// mapping into guest-physical space have been attempted (design
    /// §7). Replacing an already-set page would silently orphan the
    /// previous `MmapRegion`'s munmap until this device drops; the
    /// device manager only calls this once per instance.
    pub fn set_vintf_page(&mut self, mapping: MmapRegion) {
        self.vintf_page = Some(VintfPage::new(mapping));
    }

    /// Record the KVM memslot ids created for this instance's VINTF
    /// page aliases (design §3, §7 item 5), if any succeeded - see the
    /// `memslots` field doc comment for why no removal is needed.
    /// Called once by the device manager, with zero, one or two ids
    /// depending on how many of the two independent memslot attempts
    /// succeeded.
    pub fn set_memslots(&mut self, memslots: Vec<u32>) {
        self.memslots = memslots;
    }

    /// The KVM memslot ids recorded by `set_memslots`, for diagnostics.
    pub fn memslots(&self) -> &[u32] {
        &self.memslots
    }

    /// Attach the vIOMMU backing this instance's hardware queues.
    /// Called once by the device manager (mirrors
    /// `Smmuv3Accel::set_viommu`); a VCMDQ BASE write before this is
    /// called only latches the register cache.
    pub fn set_viommu(&mut self, viommu: Arc<IommufdVIommu>) {
        self.viommu = Some(viommu);
    }

    /// Destroy every allocated hardware queue, in descending index
    /// order. The kernel requires descending-index destroy (a lower-
    /// index VCMDQ may be a dependency of a higher-index one), but
    /// Rust drops `Vec` elements in ascending order, so this runs
    /// explicitly instead of relying on `self.vcmdq`'s own drop glue.
    /// Shared by `Drop` and `release_accel`, both of which must destroy
    /// every hardware queue before `self.viommu` (the queues' owning
    /// vIOMMU) is torn down.
    fn destroy_hw_queues(&mut self) {
        for index in (0..self.vcmdq.len()).rev() {
            self.vcmdq[index] = None;
        }
    }

    /// Ordered iommufd teardown for this instance's hardware queues and
    /// vIOMMU: the descending-index hardware-queue destroy above, then
    /// `self.viommu = None`. Meant to be run explicitly (from
    /// `DeviceManager::drop`, alongside `Smmuv3Accel::teardown`) while
    /// every VFIO cdev is still open. Without the second half,
    /// `cmdqv_devices` keeps this instance's vIOMMU alive past
    /// `Smmuv3Accel::teardown`'s own vIOMMU drop, and the S2 HWPT's
    /// EBUSY this recipe fixes survives on CMDQV hosts.
    pub fn release_accel(&mut self) {
        self.destroy_hw_queues();
        // The VINTF page0 mmap must go BEFORE the vIOMMU reference, and
        // dropping the `Arc` alone is not enough. That mapping is over the
        // iommufd fd, and the kernel takes an object reference for the
        // lifetime of the vma: `iommufd_fops_mmap` does
        // `refcount_inc_not_zero(&immap->owner->users)` and only
        // `iommufd_fops_vma_close` — i.e. munmap — releases it (v6.17
        // drivers/iommu/iommufd/main.c:581 and :551-556). The owner here is
        // the vIOMMU (`iommufd_viommu_alloc_mmap` in tegra241-cmdqv).
        // `IOMMU_DESTROY` uses `refcount_dec_if_one`, so while this page
        // stays mapped the vIOMMU destroy returns EBUSY, and the S2 HWPT and
        // IOAS then fail behind it. Leaving it to field-drop glue (which runs
        // after `DeviceManager::drop`'s body) would make a cmdqv=on shutdown
        // report THREE destroy failures where the unfixed code reported two.
        self.vintf_page = None;
        self.viommu = None;
    }

    /// Allocate (or replace) the hardware queue backing VCMDQ `index`
    /// from its currently-latched BASE (address, size). Mirrors
    /// `tegra241_cmdqv_setup_vcmdq()`: no-op (guarded, logged) unless
    /// a vIOMMU is attached, the negotiated size is non-zero, and the
    /// queue memory range is guest RAM; drops any queue already
    /// allocated at this index first (the kernel does not allow two
    /// live hardware queues at the same `(viommu, index)`).
    fn setup_vcmdq(&mut self, index: usize) {
        // ADDR occupies BASE_L bits [31:5] and BASE_H bits [15:0]
        // (shifted by 32) of the composed 64-bit register;
        // LOG2SIZE occupies BASE_L bits [4:0] (design §4).
        const BASE_ADDR_MASK: u64 = 0x0000_ffff_ffff_ffe0;
        let base = self.vcmdq_base[index];
        let addr = base & BASE_ADDR_MASK;
        let size = 1u64 << ((base & 0x1f) + 4);

        // A raw zero write clears the queue registration (the guest
        // driver zeroes BASE while tearing a VCMDQ down or before
        // programming it); drop any live hardware queue and stop.
        if base == 0 {
            if self.vcmdq[index].take().is_some() {
                debug!(
                    "Tegra241 CMDQV {}: VCMDQ{index} cleared (BASE zeroed)",
                    self.name
                );
            }
            return;
        }

        let Some(viommu) = self.viommu.clone() else {
            debug!(
                "Tegra241 CMDQV {}: VCMDQ{index} BASE latched (addr={addr:#x}, size={size:#x}), \
                 no vIOMMU attached yet",
                self.name
            );
            return;
        };

        // Bound the guest-programmed size (see `size_fits_cap`); note
        // `size` is always a power of two (`1u64 << n`), so it is never
        // zero - unlike the dead `size == 0` guard this replaces.
        let cap = self.queue_size_cap;
        if !size_fits_cap(size, cap) {
            warn!(
                "Tegra241 CMDQV {}: VCMDQ{index} negotiated size {size:#x} exceeds the \
                 {cap:#x} backing-granule cap; not allocating (existing queue, if any, is kept)",
                self.name
            );
            return;
        }

        if !addr_in_guest_ram(&self.mem, addr, size) {
            warn!(
                "Tegra241 CMDQV {}: VCMDQ{index} BASE (addr={addr:#x}, size={size:#x}) \
                 is not entirely guest RAM",
                self.name
            );
            return;
        }

        // Drop any queue already allocated at this index (its Drop
        // impl issues IOMMU_DESTROY) before allocating the new one.
        self.vcmdq[index] = None;

        match viommu.allocate_hw_queue(index as u32, addr, size) {
            Ok(hw_queue) => {
                info!(
                    "Tegra241 CMDQV {}: allocated VCMDQ{index} (addr={addr:#x}, size={size:#x})",
                    self.name
                );
                self.vcmdq[index] = Some(hw_queue);
            }
            Err(e) => warn!(
                "Tegra241 CMDQV {}: failed to allocate VCMDQ{index} \
                 (addr={addr:#x}, size={size:#x}): {e:?}",
                self.name
            ),
        }
    }

    /// Byte offset of VCMDQ `index`'s register `reg` within the
    /// VINTF page0 mmap (same layout as the trapped VCMDQ page0
    /// registers it backs).
    fn vintf_page_offset(index: usize, reg: u64) -> usize {
        (index as u64 * offset::VCMDQ_STRIDE + reg) as usize
    }

    fn read_vcmdq_page0(&mut self, index: usize, reg: u64) -> u64 {
        // If the host page is mapped, it is the ground truth for the
        // six known registers: refresh the cache from it before
        // returning, exactly as QEMU's fallback path does. Only
        // reached at all when the memslot aliasing this same page into
        // guest-physical space could not be created (design §7) -
        // otherwise the guest's accesses never trap here in the first
        // place.
        //
        // The page read must stay INSIDE each known-register arm below
        // rather than being issued once up front for any decoded
        // `reg`: an undefined offset within the `0x80` stride (e.g.
        // `0x18`, between GERRORN and the next VCMDQ's stride) has no
        // shadow field to refresh and must never touch live hardware
        // on the VMM's behalf - the guest never asked for it.
        (match reg {
            offset::VCMDQ_CONS_INDX => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_cons_indx[index] =
                        page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_cons_indx[index]
            }
            offset::VCMDQ_PROD_INDX => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_prod_indx[index] =
                        page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_prod_indx[index]
            }
            offset::VCMDQ_CONFIG => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_config[index] = page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_config[index]
            }
            offset::VCMDQ_STATUS => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_status[index] = page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_status[index]
            }
            offset::VCMDQ_GERROR => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_gerror[index] = page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_gerror[index]
            }
            offset::VCMDQ_GERRORN => {
                if let Some(page) = &self.vintf_page {
                    self.vcmdq_gerrorn[index] = page.read_u32(Self::vintf_page_offset(index, reg));
                }
                self.vcmdq_gerrorn[index]
            }
            _ => {
                trace!(
                    "Tegra241 CMDQV {}: read from unknown VCMDQ{index} page0 register {reg:#x}",
                    self.name
                );
                0
            }
        }) as u64
    }

    /// Mirror a page0 register write into the host VINTF page mmap,
    /// if one is set; a no-op otherwise (the register-cache-only
    /// fallback, exercised whenever no page has been mapped - always
    /// true in unit tests).
    fn mirror_vcmdq_page0_write(&self, index: usize, reg: u64, value: u32) {
        if let Some(page) = &self.vintf_page {
            page.write_u32(Self::vintf_page_offset(index, reg), value);
        }
    }

    fn write_vcmdq_page0(&mut self, index: usize, reg: u64, data: &[u8]) {
        let value = value32(data);
        match reg {
            offset::VCMDQ_CONS_INDX => {
                self.mirror_vcmdq_page0_write(index, reg, value);
                self.vcmdq_cons_indx[index] = value;
            }
            offset::VCMDQ_PROD_INDX => {
                self.mirror_vcmdq_page0_write(index, reg, value);
                self.vcmdq_prod_indx[index] = value;
            }
            offset::VCMDQ_CONFIG => {
                if self.vintf_page.is_some() {
                    // The real hardware CONFIG register is live once
                    // mapped: it derives its own STATUS.CMDQ_EN_OK,
                    // which the next read refreshes from the page.
                    self.mirror_vcmdq_page0_write(index, reg, value);
                } else if value & VCMDQ_CONFIG_CMDQ_EN_MASK != 0 {
                    self.vcmdq_status[index] |= VCMDQ_STATUS_CMDQ_EN_OK_MASK;
                } else {
                    self.vcmdq_status[index] &= !VCMDQ_STATUS_CMDQ_EN_OK_MASK;
                }
                self.vcmdq_config[index] = value;
            }
            offset::VCMDQ_GERRORN => {
                self.mirror_vcmdq_page0_write(index, reg, value);
                self.vcmdq_gerrorn[index] = value;
            }
            // GERROR and STATUS are read-only from the guest's
            // perspective (no case in tegra241_cmdqv_write_vcmdq()).
            _ => trace!(
                "Tegra241 CMDQV {}: write to unknown or read-only VCMDQ{index} page0 register {reg:#x}",
                self.name
            ),
        }
    }

    fn read_vcmdq_page1(&self, index: usize, reg: u64) -> u64 {
        match reg {
            offset::VCMDQ_BASE_L => self.vcmdq_base[index],
            offset::VCMDQ_BASE_H => self.vcmdq_base[index] >> 32,
            offset::VCMDQ_CONS_INDX_BASE_DRAM_L => self.vcmdq_cons_indx_base[index],
            offset::VCMDQ_CONS_INDX_BASE_DRAM_H => self.vcmdq_cons_indx_base[index] >> 32,
            _ => {
                trace!(
                    "Tegra241 CMDQV {}: read from unknown VCMDQ{index} page1 register {reg:#x}",
                    self.name
                );
                0
            }
        }
    }

    fn write_vcmdq_page1(&mut self, index: usize, reg: u64, data: &[u8]) {
        // BASE_L and CONS_INDX_BASE_DRAM_L are the "low" half of a
        // conceptual 64-bit register; the guest may write either half
        // independently (4-byte access) or the whole register in one
        // go (8-byte `writeq`, handled natively here - splitting it
        // into two 32-bit writes would fire a spurious BASE-change
        // trigger with a half-formed address once hardware queue
        // allocation is wired up).
        match reg {
            offset::VCMDQ_BASE_L => {
                if data.len() == 8 {
                    self.vcmdq_base[index] = value64(data);
                } else {
                    self.vcmdq_base[index] =
                        (self.vcmdq_base[index] & 0xffff_ffff_0000_0000) | value32(data) as u64;
                }
                self.setup_vcmdq(index);
            }
            offset::VCMDQ_BASE_H => {
                self.vcmdq_base[index] = (self.vcmdq_base[index] & 0x0000_0000_ffff_ffff)
                    | ((value32(data) as u64) << 32);
                self.setup_vcmdq(index);
            }
            offset::VCMDQ_CONS_INDX_BASE_DRAM_L => {
                if data.len() == 8 {
                    self.vcmdq_cons_indx_base[index] = value64(data);
                } else {
                    self.vcmdq_cons_indx_base[index] = (self.vcmdq_cons_indx_base[index]
                        & 0xffff_ffff_0000_0000)
                        | value32(data) as u64;
                }
            }
            offset::VCMDQ_CONS_INDX_BASE_DRAM_H => {
                self.vcmdq_cons_indx_base[index] = (self.vcmdq_cons_indx_base[index]
                    & 0x0000_0000_ffff_ffff)
                    | ((value32(data) as u64) << 32);
            }
            _ => trace!(
                "Tegra241 CMDQV {}: write to unknown VCMDQ{index} page1 register {reg:#x}",
                self.name
            ),
        }
    }

    fn read_value(&mut self, offset: u64) -> u64 {
        match decode(offset) {
            Some(Reg::Config) => self.config as u64,
            Some(Reg::Param) => self.param as u64,
            Some(Reg::Status) => self.status as u64,
            Some(Reg::ViErrMap(i)) => self.vi_err_map[i] as u64,
            Some(Reg::ViIntMask(i)) => self.vi_int_mask[i] as u64,
            Some(Reg::CmdqErrMap(i)) => self.cmdq_err_map[i] as u64,
            Some(Reg::CmdqAllocMap(i)) => self.cmdq_alloc_map[i] as u64,
            Some(Reg::Vintf0Config) => self.vintf_config as u64,
            Some(Reg::Vintf0Status) => self.vintf_status as u64,
            Some(Reg::Vintf0LvcmdqErrMap(i)) => self.vintf_lvcmdq_err_map[i] as u64,
            Some(Reg::VcmdqPage0 { index, reg }) => self.read_vcmdq_page0(index, reg),
            Some(Reg::VcmdqPage1 { index, reg }) => self.read_vcmdq_page1(index, reg),
            None => {
                self.log_unhandled(offset, "read");
                0
            }
        }
    }

    fn write_value(&mut self, offset: u64, data: &[u8]) {
        match decode(offset) {
            Some(Reg::Config) => {
                let value = value32(data);
                self.config = value;
                if value & CONFIG_CMDQV_EN_MASK != 0 {
                    self.status |= STATUS_CMDQV_ENABLED_MASK;
                } else {
                    self.status &= !STATUS_CMDQV_ENABLED_MASK;
                }
            }
            // PARAM, STATUS, VI_ERR_MAP and CMDQ_ERR_MAP are
            // read-only caches (no case in tegra241_cmdqv_write()).
            Some(Reg::Param | Reg::Status | Reg::ViErrMap(_) | Reg::CmdqErrMap(_)) => {
                self.log_unhandled(offset, "write");
            }
            Some(Reg::ViIntMask(i)) => self.vi_int_mask[i] = value32(data),
            Some(Reg::CmdqAllocMap(i)) => self.cmdq_alloc_map[i] = value32(data),
            Some(Reg::Vintf0Config) => {
                // Strip HYP_OWN from the guest write: the guest kernel
                // is not the hypervisor (tegra241-cmdqv.c:411-413).
                let value = value32(data) & !VINTF_HYP_OWN_MASK;
                self.vintf_config = value;
                if value & VINTF_ENABLE_MASK != 0 {
                    self.vintf_status |= VINTF_STATUS_ENABLE_OK_MASK;
                } else {
                    self.vintf_status &= !VINTF_STATUS_ENABLE_OK_MASK;
                }
            }
            // VINTF0_STATUS and the LVCMDQ error maps are read-only.
            Some(Reg::Vintf0Status | Reg::Vintf0LvcmdqErrMap(_)) => {
                self.log_unhandled(offset, "write");
            }
            Some(Reg::VcmdqPage0 { index, reg }) => self.write_vcmdq_page0(index, reg, data),
            Some(Reg::VcmdqPage1 { index, reg }) => self.write_vcmdq_page1(index, reg, data),
            None => self.log_unhandled(offset, "write"),
        }
    }

    fn log_unhandled(&self, offset: u64, access: &str) {
        if offset >= offset::SPAN_LIMIT {
            warn!(
                "Tegra241 CMDQV {}: {access} offset {offset:#x} exceeds the {:#x} span",
                self.name,
                offset::SPAN_LIMIT
            );
        } else {
            trace!(
                "Tegra241 CMDQV {}: unhandled {access} access at offset {offset:#x}",
                self.name
            );
        }
    }
}

impl BusDevice for Tegra241Cmdqv {
    fn read(&mut self, _base: u64, offset: u64, data: &mut [u8]) {
        match data.len() {
            4 | 8 => {
                let value = self.read_value(offset);
                data.copy_from_slice(&value.to_le_bytes()[..data.len()]);
            }
            _ => warn!(
                "Tegra241 CMDQV {}: unsupported read size {} at offset {offset:#x}",
                self.name,
                data.len()
            ),
        }
    }

    fn write(&mut self, _base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        match data.len() {
            4 | 8 => self.write_value(offset, data),
            _ => warn!(
                "Tegra241 CMDQV {}: unsupported write size {} at offset {offset:#x}",
                self.name,
                data.len()
            ),
        }
        None
    }
}

impl Drop for Tegra241Cmdqv {
    /// Destroy allocated hardware queues in descending index order
    /// before the vIOMMU they belong to (held in `self.viommu`) is
    /// dropped. The kernel requires descending-index destroy (a
    /// lower-index VCMDQ may be a dependency of a higher-index one),
    /// but Rust drops struct fields in declaration order (ascending
    /// for `self.vcmdq`, and before `self.viommu` regardless since it
    /// is declared first) - wrong on both counts, so this runs
    /// explicitly ahead of the automatic field drop glue.
    fn drop(&mut self) {
        self.destroy_hw_queues();
    }
}

impl Snapshottable for Tegra241Cmdqv {
    fn id(&self) -> String {
        self.name.clone()
    }

    /// Block snapshot, and therefore live migration and save/restore,
    /// for the same reason as `SMMUv3`'s: the load-bearing state
    /// (the kernel's hardware queue objects, the vIOMMU) lives in the
    /// host kernel, where it cannot be migrated at all.
    fn snapshot(&mut self) -> Result<Snapshot, MigratableError> {
        Err(MigratableError::Snapshot(anyhow!(
            "Tegra241 CMDQV ({}) does not support snapshot, restore or live migration",
            self.name
        )))
    }
}
impl Pausable for Tegra241Cmdqv {}
impl Transportable for Tegra241Cmdqv {}
impl Migratable for Tegra241Cmdqv {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mem() -> GuestMemoryAtomic<GuestMemoryMmap<AtomicBitmap>> {
        let mem: GuestMemoryMmap<AtomicBitmap> =
            GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x1000_0000)]).unwrap();
        GuestMemoryAtomic::new(mem)
    }

    fn dev() -> Tegra241Cmdqv {
        Tegra241Cmdqv::new("vcmdqv_0".to_string(), 0, 100, test_mem())
    }

    /// Create a page-backed `MmapRegion` for exercising the VINTF
    /// page0 mirror fallback. Real hardware provides this mapping
    /// through iommufd; a memfd stands in for it here.
    fn test_vintf_mapping(len: u64) -> MmapRegion {
        use std::ffi::CString;
        use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};

        // SAFETY: FFI call; memfd_create has no preconditions beyond
        // a valid name pointer and flags.
        let raw_fd = unsafe {
            libc::syscall(
                libc::SYS_memfd_create,
                CString::new("cmdqv-test-vintf").unwrap().as_ptr(),
                0,
            )
        };
        assert!(raw_fd >= 0, "memfd_create failed");
        // SAFETY: `raw_fd` was just returned by memfd_create above, is
        // a valid fd, and we exclusively own it from this point on.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd as i32) };
        // SAFETY: FFI call with a valid, owned fd.
        let ret = unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) };
        assert_eq!(ret, 0, "ftruncate failed");
        MmapRegion::mmap(len, libc::PROT_READ | libc::PROT_WRITE, fd.as_fd(), 0, 0).unwrap()
    }

    #[test]
    fn test_resets() {
        let d = dev();
        assert_eq!(d.config, CONFIG_RESET);
        assert_eq!(d.config, 0x0002_0403);
        assert_eq!(d.param, PARAM_RESET);
        assert_eq!(d.param, 0x0000_4011);
        assert_eq!(d.status, STATUS_RESET);
        assert_eq!(
            d.status & STATUS_CMDQV_ENABLED_MASK,
            STATUS_CMDQV_ENABLED_MASK
        );
    }

    #[test]
    fn test_decode_scalar_registers() {
        assert_eq!(decode(0x0000), Some(Reg::Config));
        assert_eq!(decode(0x0004), Some(Reg::Param));
        assert_eq!(decode(0x0008), Some(Reg::Status));
        assert_eq!(decode(0x1000), Some(Reg::Vintf0Config));
        assert_eq!(decode(0x1004), Some(Reg::Vintf0Status));
    }

    #[test]
    fn test_decode_array_registers() {
        assert_eq!(decode(0x0014), Some(Reg::ViErrMap(0)));
        assert_eq!(decode(0x0018), Some(Reg::ViErrMap(1)));
        assert_eq!(decode(0x001c), Some(Reg::ViIntMask(0)));
        assert_eq!(decode(0x0020), Some(Reg::ViIntMask(1)));
        assert_eq!(decode(0x0024), Some(Reg::CmdqErrMap(0)));
        assert_eq!(decode(0x0030), Some(Reg::CmdqErrMap(3)));
        assert_eq!(decode(0x0200), Some(Reg::CmdqAllocMap(0)));
        assert_eq!(decode(0x03fc), Some(Reg::CmdqAllocMap(127)));
        assert_eq!(decode(0x10c0), Some(Reg::Vintf0LvcmdqErrMap(0)));
        assert_eq!(decode(0x10cc), Some(Reg::Vintf0LvcmdqErrMap(3)));
    }

    #[test]
    fn test_decode_vcmdq_pages() {
        // Index 0.
        assert_eq!(
            decode(0x10000),
            Some(Reg::VcmdqPage0 {
                index: 0,
                reg: offset::VCMDQ_CONS_INDX
            })
        );
        assert_eq!(
            decode(0x10014),
            Some(Reg::VcmdqPage0 {
                index: 0,
                reg: offset::VCMDQ_GERRORN
            })
        );
        assert_eq!(
            decode(0x20000),
            Some(Reg::VcmdqPage1 {
                index: 0,
                reg: offset::VCMDQ_BASE_L
            })
        );

        // Index 1: one 0x80 stride in.
        assert_eq!(
            decode(0x10080),
            Some(Reg::VcmdqPage0 {
                index: 1,
                reg: offset::VCMDQ_CONS_INDX
            })
        );
        assert_eq!(
            decode(0x20084),
            Some(Reg::VcmdqPage1 {
                index: 1,
                reg: offset::VCMDQ_BASE_H
            })
        );

        // Index 127: the last architectural index.
        assert_eq!(
            decode(0x10000 + 127 * 0x80),
            Some(Reg::VcmdqPage0 {
                index: 127,
                reg: offset::VCMDQ_CONS_INDX
            })
        );

        // Beyond index 127 the page still has room (0x10000 bytes /
        // 0x80 stride = 2048 slots) but no register is defined there.
        assert_eq!(decode(0x10000 + 128 * 0x80), None);
    }

    #[test]
    fn test_decode_vi_vcmdq_fold() {
        // The VI_VCMDQ aliases (+0x20000) decode identically to their
        // native VCMDQ counterparts.
        assert_eq!(decode(0x30000), decode(0x10000));
        assert_eq!(decode(0x30014), decode(0x10014));
        assert_eq!(decode(0x40000), decode(0x20000));
        assert_eq!(decode(0x3fffc), decode(0x1fffc));
        assert_eq!(
            decode(0x30080),
            Some(Reg::VcmdqPage0 {
                index: 1,
                reg: offset::VCMDQ_CONS_INDX
            })
        );
    }

    #[test]
    fn test_decode_gaps_and_out_of_range() {
        // Gaps between defined registers.
        assert_eq!(decode(0x000c), None);
        assert_eq!(decode(0x0034), None);
        assert_eq!(decode(0x0400), None);
        // At and beyond the 0x50000 span limit.
        assert_eq!(decode(0x50000), None);
        assert_eq!(decode(0x50004), None);
        assert_eq!(decode(0xffff_ffff), None);
    }

    #[test]
    fn test_config_write_mirrors_status() {
        let mut d = dev();

        // Disable CMDQV_EN: STATUS.CMDQV_ENABLED must clear.
        d.write(0, offset::CONFIG, &0u32.to_le_bytes());
        assert_eq!(d.config, 0);
        assert_eq!(d.status & STATUS_CMDQV_ENABLED_MASK, 0);

        // Re-enable: STATUS.CMDQV_ENABLED must set again.
        d.write(0, offset::CONFIG, &CONFIG_CMDQV_EN_MASK.to_le_bytes());
        assert_eq!(
            d.status & STATUS_CMDQV_ENABLED_MASK,
            STATUS_CMDQV_ENABLED_MASK
        );
    }

    #[test]
    fn test_vintf_config_strips_hyp_own() {
        let mut d = dev();

        // The guest sets ENABLE and (incorrectly) HYP_OWN.
        let value = VINTF_ENABLE_MASK | VINTF_HYP_OWN_MASK;
        d.write(0, offset::VINTF0_CONFIG, &value.to_le_bytes());

        // HYP_OWN must never stick; ENABLE must, and STATUS.ENABLE_OK
        // mirrors it.
        assert_eq!(d.vintf_config, VINTF_ENABLE_MASK);
        assert_eq!(d.vintf_config & VINTF_HYP_OWN_MASK, 0);
        assert_eq!(
            d.vintf_status & VINTF_STATUS_ENABLE_OK_MASK,
            VINTF_STATUS_ENABLE_OK_MASK
        );

        // Disabling clears STATUS.ENABLE_OK.
        d.write(0, offset::VINTF0_CONFIG, &0u32.to_le_bytes());
        assert_eq!(d.vintf_status & VINTF_STATUS_ENABLE_OK_MASK, 0);
    }

    #[test]
    fn test_base_composition_4_byte_writes() {
        let mut d = dev();
        let base_l_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_L;
        let base_h_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_H;

        // ADDR bits [26:0] in BASE_L (shifted by 5), LOG2SIZE in bits
        // [4:0]; ADDR bits [15:0] in BASE_H (shifted by 32).
        let base_l: u32 = (0x1234_5678 << 5) | 0xf; // arbitrary low bits + log2size=15
        let base_h: u32 = 0xbeef;

        d.write(0, base_l_off, &base_l.to_le_bytes());
        d.write(0, base_h_off, &base_h.to_le_bytes());

        let expected = ((base_h as u64) << 32) | base_l as u64;
        assert_eq!(d.vcmdq_base[0], expected);

        // Reading back BASE_L/BASE_H recomposes the same value.
        let mut low = [0u8; 4];
        d.read(0, base_l_off, &mut low);
        assert_eq!(u32::from_le_bytes(low), base_l);
        let mut high = [0u8; 4];
        d.read(0, base_h_off, &mut high);
        assert_eq!(u32::from_le_bytes(high), base_h);
    }

    #[test]
    fn test_base_composition_8_byte_write_identical_to_4_byte() {
        let mut two_writes = dev();
        let base_l_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_L;
        let base_h_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_H;
        let base_l: u32 = (0x0abc_def0 << 5) | 0x1f;
        let base_h: u32 = 0x0007;
        two_writes.write(0, base_l_off, &base_l.to_le_bytes());
        two_writes.write(0, base_h_off, &base_h.to_le_bytes());

        let mut one_write = dev();
        let full: u64 = ((base_h as u64) << 32) | base_l as u64;
        one_write.write(0, base_l_off, &full.to_le_bytes());

        assert_eq!(one_write.vcmdq_base[0], two_writes.vcmdq_base[0]);
        assert_eq!(one_write.vcmdq_base[0], full);
    }

    #[test]
    fn test_base_l_4_byte_write_preserves_prior_high_half() {
        let mut d = dev();
        let base_l_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_L;
        let base_h_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_H;

        d.write(0, base_h_off, &0xdead_u32.to_le_bytes());
        d.write(0, base_l_off, &0x0000_0001_u32.to_le_bytes());

        assert_eq!(d.vcmdq_base[0], (0xdead_u64 << 32) | 1);
    }

    #[test]
    fn test_cons_indx_base_dram_composition() {
        let mut d = dev();
        let low_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_CONS_INDX_BASE_DRAM_L;
        let high_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_CONS_INDX_BASE_DRAM_H;
        let full: u64 = 0x0000_1234_5678_9000;

        d.write(0, low_off, &full.to_le_bytes());
        assert_eq!(d.vcmdq_cons_indx_base[0], full);

        let mut low = [0u8; 4];
        d.read(0, low_off, &mut low);
        assert_eq!(u32::from_le_bytes(low), full as u32);
        let mut high = [0u8; 4];
        d.read(0, high_off, &mut high);
        assert_eq!(u32::from_le_bytes(high), (full >> 32) as u32);
    }

    #[test]
    fn test_8_byte_write_to_32_bit_register_truncates() {
        // A stray 8-byte write to a 32-bit-only register (e.g. CONFIG)
        // must use only the low 32 bits, matching the C-level
        // semantics of tegra241_cmdqv_write() (implicit truncation on
        // assignment to a uint32_t field) - not spill into the next
        // register the way SMMUv3's generic split would.
        let mut d = dev();
        let value: u64 = CONFIG_CMDQV_EN_MASK as u64 | (0xffff_ffffu64 << 32);
        d.write(0, offset::CONFIG, &value.to_le_bytes());
        assert_eq!(d.config, CONFIG_CMDQV_EN_MASK);
        // PARAM (the next register) must be untouched.
        assert_eq!(d.param, PARAM_RESET);
    }

    #[test]
    fn test_vcmdq_page0_config_mirrors_status() {
        let mut d = dev();
        let config_off = offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_CONFIG;
        let status_off = offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_STATUS;

        d.write(0, config_off, &VCMDQ_CONFIG_CMDQ_EN_MASK.to_le_bytes());
        let mut status = [0u8; 4];
        d.read(0, status_off, &mut status);
        assert_eq!(
            u32::from_le_bytes(status) & VCMDQ_STATUS_CMDQ_EN_OK_MASK,
            VCMDQ_STATUS_CMDQ_EN_OK_MASK
        );

        d.write(0, config_off, &0u32.to_le_bytes());
        d.read(0, status_off, &mut status);
        assert_eq!(u32::from_le_bytes(status) & VCMDQ_STATUS_CMDQ_EN_OK_MASK, 0);
    }

    #[test]
    fn test_bad_access_size_ignored() {
        let mut d = dev();
        let mut data = [0u8; 2];
        // 2-byte accesses are logged and ignored, not a panic.
        d.read(0, offset::CONFIG, &mut data);
        d.write(0, offset::CONFIG, &data);
        assert_eq!(d.config, CONFIG_RESET);
    }

    #[test]
    fn test_vintf_page_mirrors_writes() {
        let mut d = dev();
        d.set_vintf_page(test_vintf_mapping(0x10000));

        let cons_off = offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_CONS_INDX;
        d.write(0, cons_off, &0x1234u32.to_le_bytes());

        // The write landed in the mmap, not just the cache.
        assert_eq!(
            d.vintf_page.as_ref().unwrap().read_u32(0),
            0x1234,
            "write must reach the mapped page, not just the register cache"
        );

        // A guest read reflects the same value.
        let mut data = [0u8; 4];
        d.read(0, cons_off, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0x1234);
    }

    #[test]
    fn test_vintf_page_read_refreshes_cache_from_hardware() {
        let mut d = dev();
        let mapping = test_vintf_mapping(0x10000);
        let base = mapping.addr();
        d.set_vintf_page(mapping);

        // Simulate real hardware advancing PROD_INDX behind the
        // VMM's back (the scenario the memslot fast path exists for):
        // poke the mmap directly, without going through the device.
        // SAFETY: `base` points into the mapping `d` now owns
        // (`0x10000` bytes of valid memory); `VCMDQ_PROD_INDX` (4) is
        // well within it.
        unsafe {
            ptr::write_volatile(
                base.add(offset::VCMDQ_PROD_INDX as usize).cast::<u32>(),
                0xabcd,
            );
        }

        let mut data = [0u8; 4];
        d.read(
            0,
            offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_PROD_INDX,
            &mut data,
        );
        assert_eq!(
            u32::from_le_bytes(data),
            0xabcd,
            "a read must refresh the cache from the mapped page"
        );
    }

    #[test]
    fn test_vintf_page_shared_between_native_and_vi_alias() {
        let mut d = dev();
        d.set_vintf_page(test_vintf_mapping(0x10000));

        let native_off = offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_CONS_INDX;
        let alias_off = offset::VI_VCMDQ_PAGE0_BASE + offset::VCMDQ_CONS_INDX;

        d.write(0, native_off, &0x55aa_u32.to_le_bytes());

        let mut data = [0u8; 4];
        d.read(0, alias_off, &mut data);
        assert_eq!(
            u32::from_le_bytes(data),
            0x55aa,
            "the VI_VCMDQ alias must observe a write through the native VCMDQ page \
             (both fold to the same mapped page)"
        );
    }

    #[test]
    fn test_vintf_page_config_write_skips_local_status_derivation() {
        let mut d = dev();
        d.set_vintf_page(test_vintf_mapping(0x10000));

        let config_off = offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_CONFIG;
        d.write(0, config_off, &VCMDQ_CONFIG_CMDQ_EN_MASK.to_le_bytes());

        // The write reached the mapped page (hardware's CONFIG)...
        assert_eq!(
            d.vintf_page
                .as_ref()
                .unwrap()
                .read_u32(offset::VCMDQ_CONFIG as usize),
            VCMDQ_CONFIG_CMDQ_EN_MASK
        );

        // ...but the device did not fabricate STATUS.CMDQ_EN_OK
        // itself: real hardware would set it in the mapped page,
        // which this test's memfd (unlike real hardware) never does,
        // so a STATUS read must come back 0, not the locally-derived
        // value the no-mmap fallback would have produced.
        let mut status = [0u8; 4];
        d.read(
            0,
            offset::VCMDQ_PAGE0_BASE + offset::VCMDQ_STATUS,
            &mut status,
        );
        assert_eq!(u32::from_le_bytes(status), 0);
    }

    #[test]
    fn test_vintf_page_read_of_undefined_reg_skips_device_access() {
        let mut d = dev();
        let mapping = test_vintf_mapping(0x10000);
        let base = mapping.addr();
        d.set_vintf_page(mapping);

        // Poke a value into the mmap at an offset within the VCMDQ
        // page0 stride that does not decode to any of the six known
        // registers (0x18 sits between GERRORN=0x14 and the next
        // VCMDQ's stride at 0x80).
        let undefined_reg = 0x18u64;
        // SAFETY: `base` points into the mapping `d` now owns
        // (`0x10000` bytes of valid memory); `undefined_reg` (0x18) is
        // well within it.
        unsafe {
            ptr::write_volatile(base.add(undefined_reg as usize).cast::<u32>(), 0xdead_beef);
        }

        let mut data = [0u8; 4];
        d.read(0, offset::VCMDQ_PAGE0_BASE + undefined_reg, &mut data);
        assert_eq!(
            u32::from_le_bytes(data),
            0,
            "an undefined VCMDQ page0 register must return the shadow (0), not the \
             live hardware value - proving the read never touched the mapped page"
        );
    }

    #[test]
    fn test_addr_in_guest_ram() {
        let mem = test_mem();
        assert!(addr_in_guest_ram(&mem, 0, 0x1000));
        assert!(addr_in_guest_ram(&mem, 0x1000_0000 - 0x1000, 0x1000));
        // Entirely out of range.
        assert!(!addr_in_guest_ram(&mem, 0x2000_0000, 0x1000));
        // Starts in range but runs off the end.
        assert!(!addr_in_guest_ram(&mem, 0x1000_0000 - 0x100, 0x1000));
    }

    #[test]
    fn test_setup_vcmdq_without_viommu_only_latches() {
        let mut d = dev();
        let base_l_off = offset::VCMDQ_PAGE1_BASE + offset::VCMDQ_BASE_L;

        // A BASE write with no vIOMMU attached must not panic, and
        // must not allocate anything (there is nothing to allocate
        // against yet).
        d.write(0, base_l_off, &0x1000_0000_u64.to_le_bytes());
        assert!(d.vcmdq[0].is_none());
        // The composed address is still latched into the cache
        // (verified in detail by the BASE-composition tests above).
        assert_eq!(d.vcmdq_base[0], 0x1000_0000);
    }

    #[test]
    fn test_setup_vcmdq_rejects_non_ram_address() {
        // Even with the guard for "no vIOMMU attached" out of the
        // way, a BASE naming an address outside guest RAM must be
        // rejected before ever reaching the vIOMMU - exercised here
        // indirectly through `addr_in_guest_ram`, since driving
        // `setup_vcmdq`'s vIOMMU-present path needs a live kernel
        // vIOMMU that a unit test cannot construct.
        let mem = test_mem();
        let log2size = 4u64; // size = 1 << (4 + 4) = 0x100
        let addr = 0x2000_0000u64; // outside the 256 MiB test RAM
        assert!(!addr_in_guest_ram(&mem, addr, 1u64 << (log2size + 4)));
    }

    #[test]
    fn test_setup_vcmdq_rejects_oversized_queue() {
        // As with `test_setup_vcmdq_rejects_non_ram_address`, driving
        // `setup_vcmdq`'s vIOMMU-present path (where an oversized BASE
        // write actually skips `allocate_hw_queue`, leaving any existing
        // queue alone) needs a live kernel vIOMMU that a unit test
        // cannot construct - exercised here indirectly through
        // `size_fits_cap`.
        let host_page = 0x1_0000u64; // 64 KiB, e.g. GH200.
        // log2size = 16 -> size = 1 << 20 (1 MiB): exceeds one 64 KiB
        // host page and must be rejected.
        assert!(!size_fits_cap(1u64 << 20, host_page));
        // A queue that exactly fits the host page is accepted.
        assert!(size_fits_cap(host_page, host_page));
        // A queue smaller than the host page is accepted.
        assert!(size_fits_cap(host_page / 2, host_page));
    }

    #[test]
    fn test_queue_size_cap_follows_the_backing_granule() {
        // With guest RAM on 512 MiB hugepages, IDR1.CMDQS is no longer
        // clamped to the host page, so Linux sizes each VCMDQ up to
        // 2^19 entries (8 MiB). The device must then accept that size:
        // a cap left at the host page refuses every VCMDQ and the
        // guest's CMD_SYNCs on them time out.
        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::<AtomicBitmap>::from_ranges(&[(GuestAddress(0), 0x1000)]).unwrap(),
        );
        let mut d = Tegra241Cmdqv::new("cmdqv".to_string(), 0, 0, mem);
        let vcmdq_8mib = 1u64 << (19 + 4);
        assert!(
            !size_fits_cap(vcmdq_8mib, d.queue_size_cap),
            "host page cap"
        );
        d.set_queue_size_cap(512 << 20);
        assert!(size_fits_cap(vcmdq_8mib, d.queue_size_cap), "hugepage cap");
    }

    #[test]
    fn test_drop_destroys_queues_descending() {
        // No vIOMMU is attached in this test, so there is nothing for
        // Drop to destroy; this only exercises that dropping a device
        // with an empty `vcmdq` table (the common case: most
        // instances never get a VFIO device assigned) does not panic.
        drop(dev());
    }

    #[test]
    fn test_snapshot_is_blocked() {
        let mut d = dev();
        assert!(matches!(d.snapshot(), Err(MigratableError::Snapshot(_))));
    }

    #[test]
    fn test_irq_and_enabled() {
        let mut d = dev();
        assert_eq!(d.irq(), 100);
        // A freshly-created instance is never live: setup hasn't run.
        assert!(!d.is_enabled());
        d.set_enabled(true);
        assert!(d.is_enabled());
        d.set_enabled(false);
        assert!(!d.is_enabled());
    }
}
