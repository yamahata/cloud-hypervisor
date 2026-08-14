// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::os::fd::BorrowedFd;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::{cmp, io, result};

use anyhow::{Context, anyhow};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use byteorder::{ByteOrder, LittleEndian};
use hypervisor::HypervisorVmError;
use libc::{_SC_PAGESIZE, sysconf};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use vfio_bindings::bindings::vfio::{
    vfio_device_mig_state_VFIO_DEVICE_STATE_ERROR as VFIO_DEV_STATE_ERROR,
    vfio_device_mig_state_VFIO_DEVICE_STATE_PRE_COPY as VFIO_DEV_STATE_PRE_COPY,
    vfio_device_mig_state_VFIO_DEVICE_STATE_PRE_COPY_P2P as VFIO_DEV_STATE_PRE_COPY_P2P,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RESUMING as VFIO_DEV_STATE_RESUMING,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RUNNING as VFIO_DEV_STATE_RUNNING,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RUNNING_P2P as VFIO_DEV_STATE_RUNNING_P2P,
    vfio_device_mig_state_VFIO_DEVICE_STATE_STOP as VFIO_DEV_STATE_STOP,
    vfio_device_mig_state_VFIO_DEVICE_STATE_STOP_COPY as VFIO_DEV_STATE_STOP_COPY, *,
};
use vfio_ioctls::{
    DmaLoggingRange, VfioDevice, VfioIrq, VfioOps, VfioRegionInfoCap, VfioRegionSparseMmapArea,
};
use vm_allocator::page_size::{
    align_page_size_down, align_page_size_up, get_page_size, is_4k_aligned, is_4k_multiple,
    is_page_size_aligned,
};
use vm_allocator::{AddressAllocator, MemorySlotAllocator, SystemAllocator};
use vm_device::dma_mapping::ExternalDmaMapping;
use vm_device::interrupt::{
    InterruptIndex, InterruptManager, InterruptSourceGroup, MsiIrqGroupConfig,
};
use vm_device::{BusDevice, Resource};
use vm_memory::bitmap::AtomicBitmap;
use vm_memory::{
    Address, GuestAddress, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryBackend,
    GuestMemoryRegion, GuestUsize,
};

type GuestMemoryMmap = vm_memory::GuestMemoryMmap<AtomicBitmap>;
use vm_migration::protocol::MemoryRangeTable;
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vmm_sys_util::eventfd::EventFd;

use crate::configuration::{
    COMMAND_REG, COMMAND_REG_MEMORY_SPACE_MASK, PCI_EXP_FLAGS_TYPE_MASK, PCI_EXP_FLAGS_VERS_MASK,
    PCI_EXP_FLAGS_VERS_SHIFT, PCI_EXP_LNKCAP, PCI_EXP_LNKCAP2, PCI_EXP_LNKCTL, PCI_EXP_LNKCTL2,
    PCI_EXP_TYPE_RC_END, PCI_EXT_CAP_ALIGN, PCI_EXT_CAP_NEXT_MASK, PCI_EXT_CAP_NEXT_SHIFT,
    PCIE_CONFIG_SPACE_SIZE,
};
use crate::mmap::MmapRegion;
use crate::msi::{MSI_CONFIG_ID, MsiConfigState};
use crate::msix::{MaybeMutInterruptSourceGroup, MsixConfigState};
use crate::{
    BarReprogrammingParams, MSIX_CONFIG_ID, MSIX_TABLE_ENTRY_SIZE, MsiCap, MsiConfig, MsixCap,
    MsixConfig, PCI_CONFIGURATION_ID, PciBarConfiguration, PciBarPrefetchable, PciBarRegionType,
    PciBdf, PciCapabilityId, PciClassCode, PciConfiguration, PciDevice, PciDeviceError,
    PciExpressCapability, PciExpressCapabilityId, PciHeaderType, PciSubclass,
    msi_num_enabled_vectors,
};

pub(crate) const VFIO_COMMON_ID: &str = "vfio_common";
pub(crate) const VFIO_MIGRATION_ID: &str = "vfio_migration";

/// Why a clique device's stage-2 mapping attempt failed closed at a
/// site that otherwise silently leaves a BAR unmapped or partially
/// mapped.
#[derive(Debug, Error)]
pub enum CliqueSkipReason {
    #[error("excluded from the stage-2 map via x_exclude_mmap_bars")]
    UserExcludedBar,
    #[error("region is not MMAP-capable (VFIO_REGION_INFO_FLAG_MMAP unset)")]
    NotMmapCapable,
    #[error("MSI-X table/PBA region lacks VFIO_REGION_INFO_CAP_MSIX_MAPPABLE")]
    MsixNotMappable,
    #[error("sparse mmap covers only 0x{mapped:x} of 0x{size:x} bytes; a stage-2 hole remains")]
    KernelSparseHole { mapped: u64, size: u64 },
}

#[derive(Debug, Error)]
pub enum VfioPciError {
    #[error("Clique device {0} BAR {1}: {2}")]
    CliqueMappingSkipped(PciBdf, u32, CliqueSkipReason),
    #[error("Failed to create user memory region")]
    CreateUserMemoryRegion(#[source] HypervisorVmError),
    #[error("Failed to DMA map: {0} for device {1} (guest BDF: {2})")]
    DmaMap(#[source] vfio_ioctls::VfioError, PathBuf, PciBdf),
    #[error("Failed to DMA unmap: {0} for device {1} (guest BDF: {2})")]
    DmaUnmap(#[source] vfio_ioctls::VfioError, PathBuf, PciBdf),
    #[error("Failed to enable INTx")]
    EnableIntx(#[source] VfioError),
    #[error("Failed to enable MSI")]
    EnableMsi(#[source] VfioError),
    #[error("Failed to enable MSI-x")]
    EnableMsix(#[source] VfioError),
    #[error("Failed to mmap the area")]
    MmapArea,
    #[error("Failed to notifier's eventfd")]
    MissingNotifier,
    #[error(
        "Failed to DMA map BAR {bar} of device {path} (guest BDF: {bdf}) \
             into the host IOMMU address space; dma-buf layer: {dmabuf}; \
             legacy layer: {source}"
    )]
    P2pDmaMapAllLayersFailed {
        #[source]
        source: vfio_ioctls::VfioError,
        path: PathBuf,
        bdf: PciBdf,
        bar: u32,
        dmabuf: String,
    },
    #[error("Invalid region alignment")]
    RegionAlignment,
    #[error("Invalid region size")]
    RegionSize,
    #[error("Failed to retrieve MsiConfigState")]
    RetrieveMsiConfigState(#[source] anyhow::Error),
    #[error("Failed to retrieve MsixConfigState")]
    RetrieveMsixConfigState(#[source] anyhow::Error),
    #[error("Failed to retrieve PciConfigurationState")]
    RetrievePciConfigurationState(#[source] anyhow::Error),
    #[error("Failed to retrieve VfioCommonState")]
    RetrieveVfioCommonState(#[source] anyhow::Error),
    #[error("No space left for the extended capability {0:#x}")]
    ExtendedCapNoSpace(u16),
    #[error("Mismatched dwords and write masks for the extended capability {0:#x}")]
    ExtendedCapMismatchedWriteMasks(u16),
    #[error("Failed to restore VFIO migration state")]
    RestoreMigration(#[source] anyhow::Error),
}

#[derive(Copy, Clone)]
enum PciVfioSubclass {
    VfioSubclass = 0xff,
}

impl PciSubclass for PciVfioSubclass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

enum InterruptUpdateAction {
    EnableMsi,
    DisableMsi,
    EnableMsix,
    DisableMsix,
}

#[derive(Serialize, Deserialize)]
struct IntxState {
    enabled: bool,
}

pub(crate) struct VfioIntx {
    interrupt_source_group: Arc<dyn InterruptSourceGroup>,
    enabled: bool,
}

#[derive(Serialize, Deserialize)]
struct MsiState {
    cap: MsiCap,
    cap_offset: u32,
}

pub(crate) struct VfioMsi {
    pub(crate) cfg: MsiConfig,
    cap_offset: u32,
    interrupt_source_group: Arc<dyn InterruptSourceGroup>,
}

impl VfioMsi {
    fn update(&mut self, offset: u64, data: &[u8]) -> Option<InterruptUpdateAction> {
        let old_enabled = self.cfg.enabled();

        self.cfg.update(offset, data);

        let new_enabled = self.cfg.enabled();

        if !old_enabled && new_enabled {
            return Some(InterruptUpdateAction::EnableMsi);
        }

        if old_enabled && !new_enabled {
            return Some(InterruptUpdateAction::DisableMsi);
        }

        None
    }
}

#[derive(Serialize, Deserialize)]
struct MsixState {
    cap: MsixCap,
    cap_offset: u32,
    bdf: u32,
}

pub(crate) struct VfioMsix {
    pub(crate) bar: MsixConfig,
    cap: MsixCap,
    cap_offset: u32,
    interrupt_source_group: Arc<dyn InterruptSourceGroup>,
}

impl VfioMsix {
    fn update(&mut self, offset: u64, data: &[u8]) -> Option<InterruptUpdateAction> {
        let old_enabled = self.bar.enabled();

        // Update "Message Control" word
        if offset == 2 && data.len() == 2 {
            let data = LittleEndian::read_u16(data);
            self.bar.set_msg_ctl(data);
            self.cap.set_msg_ctl(data);
        } else if offset == 0 && data.len() == 4 {
            // Some guests update MSI-X control through the dword config write path.
            let data = (LittleEndian::read_u32(data) >> 16) as u16;
            self.bar.set_msg_ctl(data);
            self.cap.set_msg_ctl(data);
        }

        let new_enabled = self.bar.enabled();

        if !old_enabled && new_enabled {
            return Some(InterruptUpdateAction::EnableMsix);
        }

        if old_enabled && !new_enabled {
            return Some(InterruptUpdateAction::DisableMsix);
        }

        None
    }

    fn table_accessed(&self, bar_index: u32, offset: u64) -> bool {
        let table_offset: u64 = u64::from(self.cap.table_offset());
        let table_size: u64 = u64::from(self.cap.table_size()) * (MSIX_TABLE_ENTRY_SIZE as u64);
        let table_bir: u32 = self.cap.table_bir();

        bar_index == table_bir && offset >= table_offset && offset < table_offset + table_size
    }
}

pub(crate) struct Interrupt {
    pub(crate) intx: Option<VfioIntx>,
    pub(crate) msi: Option<VfioMsi>,
    pub(crate) msix: Option<VfioMsix>,
}

impl Interrupt {
    fn update_msi(&mut self, offset: u64, data: &[u8]) -> Option<InterruptUpdateAction> {
        if let Some(msi) = &mut self.msi {
            let action = msi.update(offset, data);
            return action;
        }

        None
    }

    fn update_msix(&mut self, offset: u64, data: &[u8]) -> Option<InterruptUpdateAction> {
        if let Some(msix) = &mut self.msix {
            let action = msix.update(offset, data);
            return action;
        }

        None
    }

    fn accessed(&self, offset: u64) -> Option<(PciCapabilityId, u64)> {
        if let Some(msi) = &self.msi
            && offset >= u64::from(msi.cap_offset)
            && offset < u64::from(msi.cap_offset) + msi.cfg.size()
        {
            return Some((
                PciCapabilityId::MessageSignalledInterrupts,
                u64::from(msi.cap_offset),
            ));
        }

        if let Some(msix) = &self.msix
            && offset == u64::from(msix.cap_offset)
        {
            return Some((PciCapabilityId::MsiX, u64::from(msix.cap_offset)));
        }

        None
    }

    fn msix_table_accessed(&self, bar_index: u32, offset: u64) -> bool {
        if let Some(msix) = &self.msix {
            return msix.table_accessed(bar_index, offset);
        }

        false
    }

    fn msix_write_table(&mut self, offset: u64, data: &[u8]) {
        if let Some(msix) = &mut self.msix {
            let offset = offset - u64::from(msix.cap.table_offset());
            msix.bar.write_table(offset, data);
        }
    }

    fn msix_read_table(&self, offset: u64, data: &mut [u8]) {
        if let Some(msix) = &self.msix {
            let offset = offset - u64::from(msix.cap.table_offset());
            msix.bar.read_table(offset, data);
        }
    }

    pub(crate) fn intx_in_use(&self) -> bool {
        if let Some(intx) = &self.intx {
            return intx.enabled;
        }

        false
    }
}

#[derive(Clone)]
pub(crate) struct UserMemoryRegion {
    pub slot: u32,
    pub start: u64,
    pub mapping: Arc<MmapRegion>,
    /// True while this region's BAR content is DMA-mapped into the host
    /// IOMMU address space (peer-to-peer DMA). Stays false when the
    /// kernel cannot map MMIO into the IOMMU backend. Tracked on the
    /// owning copy in VfioCommon::mmio_regions; clones can go stale.
    pub p2p_mapped: bool,
    /// Length actually passed to the host IOMMU. Not always
    /// `mapping.len()`: a sub-page BAR's mmap is expanded to a full host
    /// page, but a dma-buf may not exceed `pci_resource_len`.
    pub p2p_len: u64,
    /// The dma-buf exported for this area, kept alive so the mapping can be
    /// re-established on a vCPU thread, which cannot call
    /// VFIO_DEVICE_FEATURE. `None` when the legacy VA mapping is in use.
    pub dmabuf: Option<Arc<File>>,
    /// Set when a revoke edge dropped this region's mapping, so the
    /// un-revoke edge knows to re-establish it. A mapping lost to a failed
    /// BAR move is deliberately not marked, and is not retried.
    pub p2p_revoked: bool,
}

#[derive(Clone)]
pub struct MmioRegion {
    pub start: GuestAddress,
    pub length: GuestUsize,
    pub(crate) type_: PciBarRegionType,
    pub(crate) index: u32,
    pub(crate) user_memory_regions: Vec<UserMemoryRegion>,
}

impl MmioRegion {
    /// Returns true if this region has the exact same memory slots as the other region.
    pub fn has_matching_slots(&self, other: &MmioRegion) -> bool {
        self.user_memory_regions.len() == other.user_memory_regions.len()
            && self
                .user_memory_regions
                .iter()
                .all(|u| other.user_memory_regions.iter().any(|o| o.slot == u.slot))
    }
}

/// # Safety
///
/// [`Self::find_user_address`] must always either return `Err`
/// or a pointer to `size` bytes of valid memory.
unsafe trait MmioRegionRange {
    fn check_range(&self, guest_addr: u64, size: u64) -> bool;
    fn find_user_address(&self, guest_addr: u64, size: u64) -> Result<*mut u8, io::Error>;
}

// SAFETY: See the comment in `find_user_address`.
unsafe impl MmioRegionRange for Vec<MmioRegion> {
    // Check if a guest address is within the range of mmio regions
    fn check_range(&self, guest_addr: u64, size: u64) -> bool {
        let Some(guest_addr_end) = guest_addr.checked_add(size) else {
            return false;
        };
        for region in self.iter() {
            let Some(region_end) = region.start.raw_value().checked_add(region.length) else {
                return false;
            };
            if guest_addr >= region.start.raw_value() && guest_addr_end <= region_end {
                return true;
            }
        }
        false
    }

    // Locate the user region address for a guest address within all mmio regions
    fn find_user_address(&self, guest_addr: u64, size: u64) -> Result<*mut u8, io::Error> {
        for region in self.iter() {
            for user_region in region.user_memory_regions.iter() {
                let mapping: &MmapRegion = &user_region.mapping;
                let start: u64 = user_region.start;
                let len: u64 = mapping.len().try_into().unwrap();
                // See if the guest address is inside the region.
                let Some(offset_from_start) = guest_addr.checked_sub(start) else {
                    continue;
                };
                if offset_from_start >= len {
                    continue;
                }
                // A VFIO MMIO region may be backed by multiple separately mmap'd areas,
                // which may be non-contiguous. The requested range must fit within one mapping.
                if size > len - offset_from_start {
                    continue;
                }
                // SAFETY: MmapRegion guarantees that mapping.addr points to at least
                // mapping.len() bytes of valid memory. The checks above ensure that
                // offset_from_start + size stays within that mapping. Also, since
                // mapping.len() fit in usize, offset_from_start must as well, so the
                // cast is safe.
                return Ok(unsafe { mapping.addr().add(offset_from_start as usize) });
            }
        }

        Err(io::Error::other(format!(
            "unable to find user mapping for DMA range \
             (gpa 0x{guest_addr:x}, size 0x{size:x})"
        )))
    }
}

#[derive(Debug, Error)]
pub enum VfioError {
    #[error("Kernel VFIO error")]
    KernelVfio(#[source] vfio_ioctls::VfioError),
    #[error("VFIO user error")]
    VfioUser(#[source] vfio_user::Error),
    #[error("VFIO device does not support migration")]
    NoMigrationSupport,
    #[error("VFIO device reported unknown migration state {0}")]
    InvalidMigrationState(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VfioMigrationState {
    Error,
    Stop,
    Running,
    StopCopy,
    Resuming,
    RunningP2P,
    PreCopy,
    PreCopyP2P,
}

impl From<VfioMigrationState> for u32 {
    fn from(state: VfioMigrationState) -> u32 {
        match state {
            VfioMigrationState::Error => VFIO_DEV_STATE_ERROR,
            VfioMigrationState::Stop => VFIO_DEV_STATE_STOP,
            VfioMigrationState::Running => VFIO_DEV_STATE_RUNNING,
            VfioMigrationState::StopCopy => VFIO_DEV_STATE_STOP_COPY,
            VfioMigrationState::Resuming => VFIO_DEV_STATE_RESUMING,
            VfioMigrationState::RunningP2P => VFIO_DEV_STATE_RUNNING_P2P,
            VfioMigrationState::PreCopy => VFIO_DEV_STATE_PRE_COPY,
            VfioMigrationState::PreCopyP2P => VFIO_DEV_STATE_PRE_COPY_P2P,
        }
    }
}

impl TryFrom<u32> for VfioMigrationState {
    type Error = VfioError;

    fn try_from(value: u32) -> Result<Self, VfioError> {
        match value {
            VFIO_DEV_STATE_ERROR => Ok(Self::Error),
            VFIO_DEV_STATE_STOP => Ok(Self::Stop),
            VFIO_DEV_STATE_RUNNING => Ok(Self::Running),
            VFIO_DEV_STATE_STOP_COPY => Ok(Self::StopCopy),
            VFIO_DEV_STATE_RESUMING => Ok(Self::Resuming),
            VFIO_DEV_STATE_RUNNING_P2P => Ok(Self::RunningP2P),
            VFIO_DEV_STATE_PRE_COPY => Ok(Self::PreCopy),
            VFIO_DEV_STATE_PRE_COPY_P2P => Ok(Self::PreCopyP2P),
            other => Err(VfioError::InvalidMigrationState(other)),
        }
    }
}

pub(crate) trait Vfio: Send + Sync {
    fn read_config_byte(&self, offset: u32) -> u8 {
        let mut data: [u8; 1] = [0];
        self.read_config(offset, &mut data);
        data[0]
    }

    fn read_config_word(&self, offset: u32) -> u16 {
        let mut data: [u8; 2] = [0, 0];
        self.read_config(offset, &mut data);
        u16::from_le_bytes(data)
    }

    fn read_config_dword(&self, offset: u32) -> u32 {
        let mut data: [u8; 4] = [0, 0, 0, 0];
        self.read_config(offset, &mut data);
        u32::from_le_bytes(data)
    }

    fn write_config_dword(&self, offset: u32, buf: u32) {
        let data: [u8; 4] = buf.to_le_bytes();
        self.write_config(offset, &data);
    }

    fn read_config(&self, offset: u32, data: &mut [u8]) {
        self.region_read(VFIO_PCI_CONFIG_REGION_INDEX, offset.into(), data.as_mut());
    }

    fn write_config(&self, offset: u32, data: &[u8]) {
        self.region_write(VFIO_PCI_CONFIG_REGION_INDEX, offset.into(), data);
    }

    fn enable_msi(&self, fds: Vec<&EventFd>) -> Result<(), VfioError> {
        self.enable_irq(VFIO_PCI_MSI_IRQ_INDEX, fds)
    }

    fn disable_msi(&self) -> Result<(), VfioError> {
        self.disable_irq(VFIO_PCI_MSI_IRQ_INDEX)
    }

    fn enable_msix(&self, fds: Vec<&EventFd>) -> Result<(), VfioError> {
        self.enable_irq(VFIO_PCI_MSIX_IRQ_INDEX, fds)
    }

    fn disable_msix(&self) -> Result<(), VfioError> {
        self.disable_irq(VFIO_PCI_MSIX_IRQ_INDEX)
    }

    fn region_read(&self, _index: u32, _offset: u64, _data: &mut [u8]) {
        unimplemented!()
    }

    fn region_write(&self, _index: u32, _offset: u64, _data: &[u8]) {
        unimplemented!()
    }

    fn get_irq_info(&self, _irq_index: u32) -> Option<VfioIrq> {
        unimplemented!()
    }

    fn enable_irq(&self, _irq_index: u32, _event_fds: Vec<&EventFd>) -> Result<(), VfioError> {
        unimplemented!()
    }

    fn disable_irq(&self, _irq_index: u32) -> Result<(), VfioError> {
        unimplemented!()
    }

    fn unmask_irq(&self, _irq_index: u32) -> Result<(), VfioError> {
        unimplemented!()
    }

    fn migration_flags(&self) -> Result<Option<u64>, VfioError> {
        Ok(None)
    }

    fn set_migration_state(&self, _state: VfioMigrationState) -> Result<(), VfioError> {
        Err(VfioError::NoMigrationSupport)
    }

    fn read_migration_data(&self) -> Result<Vec<u8>, VfioError> {
        Err(VfioError::NoMigrationSupport)
    }

    fn write_migration_data(&self, _data: &[u8]) -> Result<(), VfioError> {
        Err(VfioError::NoMigrationSupport)
    }

    fn reset(&self) {}

    fn start_dma_logging(
        &self,
        _page_size: u64,
        _ranges: &[DmaLoggingRange],
    ) -> Result<u64, VfioError> {
        Err(VfioError::NoMigrationSupport)
    }

    fn stop_dma_logging(&self) -> Result<(), VfioError> {
        Err(VfioError::NoMigrationSupport)
    }

    fn report_dma_logging(
        &self,
        _range: DmaLoggingRange,
        _page_size: u64,
    ) -> Result<MemoryRangeTable, VfioError> {
        Err(VfioError::NoMigrationSupport)
    }
}

struct VfioDeviceWrapper {
    device: Arc<VfioDevice>,
}

impl VfioDeviceWrapper {
    fn new(device: Arc<VfioDevice>) -> Self {
        Self { device }
    }
}

impl Vfio for VfioDeviceWrapper {
    fn region_read(&self, index: u32, offset: u64, data: &mut [u8]) {
        self.device.region_read(index, data, offset);
    }

    fn region_write(&self, index: u32, offset: u64, data: &[u8]) {
        self.device.region_write(index, data, offset);
    }

    fn get_irq_info(&self, irq_index: u32) -> Option<VfioIrq> {
        self.device.get_irq_info(irq_index).copied()
    }

    fn enable_irq(&self, irq_index: u32, event_fds: Vec<&EventFd>) -> Result<(), VfioError> {
        self.device
            .enable_irq(irq_index, event_fds)
            .map_err(VfioError::KernelVfio)
    }

    fn disable_irq(&self, irq_index: u32) -> Result<(), VfioError> {
        self.device
            .disable_irq(irq_index)
            .map_err(VfioError::KernelVfio)
    }

    fn unmask_irq(&self, irq_index: u32) -> Result<(), VfioError> {
        self.device
            .unmask_irq(irq_index)
            .map_err(VfioError::KernelVfio)
    }

    fn migration_flags(&self) -> Result<Option<u64>, VfioError> {
        self.device
            .query_migration_support()
            .map_err(VfioError::KernelVfio)
    }

    fn set_migration_state(&self, state: VfioMigrationState) -> Result<(), VfioError> {
        self.device
            .set_migration_state(state.into())
            .map_err(VfioError::KernelVfio)
    }

    fn read_migration_data(&self) -> Result<Vec<u8>, VfioError> {
        self.device
            .read_migration_data_to_end()
            .map_err(VfioError::KernelVfio)
    }

    fn write_migration_data(&self, data: &[u8]) -> Result<(), VfioError> {
        self.device
            .write_migration_data(data)
            .map_err(VfioError::KernelVfio)
    }

    fn reset(&self) {
        self.device.reset();
    }

    fn start_dma_logging(
        &self,
        page_size: u64,
        ranges: &[DmaLoggingRange],
    ) -> Result<u64, VfioError> {
        self.device
            .start_dma_logging(page_size, ranges)
            .map_err(VfioError::KernelVfio)
    }

    fn stop_dma_logging(&self) -> Result<(), VfioError> {
        self.device
            .stop_dma_logging()
            .map_err(VfioError::KernelVfio)
    }

    // Wrap the kernel dirty bitmap into a MemoryRangeTable at the trait
    // boundary so callers work with guest memory ranges directly.
    fn report_dma_logging(
        &self,
        range: DmaLoggingRange,
        page_size: u64,
    ) -> Result<MemoryRangeTable, VfioError> {
        let bitmap = self
            .device
            .report_dma_logging(range, page_size)
            .map_err(VfioError::KernelVfio)?;
        Ok(MemoryRangeTable::from_dirty_bitmap(
            bitmap, range.iova, page_size,
        ))
    }
}

#[derive(Serialize, Deserialize)]
struct VfioCommonState {
    intx_state: Option<IntxState>,
    msi_state: Option<MsiState>,
    msix_state: Option<MsixState>,
    #[serde(default)]
    patches: HashMap<usize, ConfigPatch>,
}

#[derive(Serialize, Deserialize)]
struct VfioMigrationData {
    blob: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ConfigPatch {
    mask: u32,
    patch: u32,
    write_mask: u32,
}

impl ConfigPatch {
    fn write(&mut self, offset: u64, data: &[u8]) {
        let mut bytes = self.patch.to_le_bytes();

        for (i, byte) in data.iter().enumerate() {
            let Some(slot) = bytes.get_mut(offset as usize + i) else {
                break;
            };
            *slot = *byte;
        }

        let written = u32::from_le_bytes(bytes);
        self.patch = (self.patch & !self.write_mask) | (written & self.write_mask);
    }
}

pub(crate) struct VfioCommon {
    pub(crate) configuration: PciConfiguration,
    pub(crate) mmio_regions: Vec<MmioRegion>,
    pub(crate) interrupt: Interrupt,
    pub(crate) msi_interrupt_manager: Arc<dyn InterruptManager<GroupConfig = MsiIrqGroupConfig>>,
    pub(crate) legacy_interrupt_group: Option<Arc<dyn InterruptSourceGroup>>,
    pub(crate) vfio_wrapper: Arc<dyn Vfio>,
    pub(crate) patches: HashMap<usize, ConfigPatch>,
    x_nv_gpudirect_clique: Option<u8>,
    x_exclude_mmap_bars: Vec<u8>,
    pub(crate) migration_flags: Option<u64>,
    // Negotiated dirty bitmap granularity while DMA logging is active.
    dma_logging_page_size: Option<u64>,
    extended_caps: Vec<Arc<dyn PciExpressCapability + Send + Sync>>,
    // Standard-capability-list offset of the PCI Express Capability, set
    // by parse_capabilities. Used to locate the Device Control register
    // (offset+8) for the FLR hook and to gate parse_extended_capabilities
    // the same way the old pci_express_cap_found bool did.
    pcie_cap_offset: Option<u8>,
    // Standard-capability-list offset of the Power Management Capability,
    // set by parse_capabilities. Used to locate the PMCSR register
    // (offset+4) for the D3 hook and to gate parse_extended_capabilities
    // the same way the old power_management_cap_found bool did.
    pm_cap_offset: Option<u8>,
    // Bumped by reset_and_rearm() after every reset. AtomicU64 because
    // reset_and_rearm takes &self; Relaxed is sufficient because the
    // comparison against it always happens later, on the same thread
    // (VfioPciDevice::reconcile_after_reset).
    reset_generation: AtomicU64,
}

#[derive(Default)]
pub(crate) struct VfioCommonConfig {
    pub(crate) x_nv_gpudirect_clique: Option<u8>,
    pub(crate) x_exclude_mmap_bars: Vec<u8>,
    pub(crate) extended_caps: Vec<Arc<dyn PciExpressCapability + Send + Sync>>,
}

impl VfioCommon {
    pub(crate) fn new(
        msi_interrupt_manager: Arc<dyn InterruptManager<GroupConfig = MsiIrqGroupConfig>>,
        legacy_interrupt_group: Option<Arc<dyn InterruptSourceGroup>>,
        vfio_wrapper: Arc<dyn Vfio>,
        subclass: &dyn PciSubclass,
        bdf: PciBdf,
        snapshot: Option<&Snapshot>,
        config: VfioCommonConfig,
    ) -> Result<Self, VfioPciError> {
        let pci_configuration_state = vm_migration::state_from_id(snapshot, PCI_CONFIGURATION_ID)
            .map_err(|e| {
            VfioPciError::RetrievePciConfigurationState(anyhow!(
                "Failed to get PciConfigurationState from Snapshot: {e}"
            ))
        })?;

        let configuration = PciConfiguration::new(
            0,
            0,
            0,
            PciClassCode::Other,
            subclass,
            None,
            PciHeaderType::Device,
            0,
            0,
            None,
            pci_configuration_state,
        );

        let migration_flags = match vfio_wrapper.migration_flags() {
            Ok(Some(flags)) => {
                info!(
                    "VFIO device {bdf} supports migration v2 (flags=0x{flags:x}, \
                     STOP_COPY={}, P2P={}, PRE_COPY={})",
                    flags & VFIO_MIGRATION_STOP_COPY as u64 != 0,
                    flags & VFIO_MIGRATION_P2P as u64 != 0,
                    flags & VFIO_MIGRATION_PRE_COPY as u64 != 0,
                );
                Some(flags)
            }
            Ok(None) => {
                debug!("VFIO device {bdf} does not support migration v2");
                None
            }
            Err(e) => {
                debug!("VFIO device {bdf} migration probe failed, treating as non-migratable: {e}");
                None
            }
        };

        let mut vfio_common = VfioCommon {
            mmio_regions: Vec::new(),
            configuration,
            interrupt: Interrupt {
                intx: None,
                msi: None,
                msix: None,
            },
            msi_interrupt_manager,
            legacy_interrupt_group,
            vfio_wrapper,
            patches: HashMap::new(),
            x_nv_gpudirect_clique: config.x_nv_gpudirect_clique,
            x_exclude_mmap_bars: config.x_exclude_mmap_bars,
            migration_flags,
            dma_logging_page_size: None,
            extended_caps: config.extended_caps,
            pcie_cap_offset: None,
            pm_cap_offset: None,
            reset_generation: AtomicU64::new(0),
        };

        let state: Option<VfioCommonState> = snapshot
            .as_ref()
            .map(|s| s.to_state())
            .transpose()
            .map_err(|e| {
                VfioPciError::RetrieveVfioCommonState(anyhow!(
                    "Failed to get VfioCommonState from Snapshot: {e}"
                ))
            })?;
        let msi_state = vm_migration::state_from_id(snapshot, MSI_CONFIG_ID).map_err(|e| {
            VfioPciError::RetrieveMsiConfigState(anyhow!(
                "Failed to get MsiConfigState from Snapshot: {e}"
            ))
        })?;
        let msix_state = vm_migration::state_from_id(snapshot, MSIX_CONFIG_ID).map_err(|e| {
            VfioPciError::RetrieveMsixConfigState(anyhow!(
                "Failed to get MsixConfigState from Snapshot: {e}"
            ))
        })?;

        if let Some(state) = state.as_ref() {
            let mig: Option<VfioMigrationData> =
                vm_migration::state_from_id(snapshot, VFIO_MIGRATION_ID).map_err(|e| {
                    VfioPciError::RestoreMigration(anyhow!(
                        "Failed to get VfioMigrationData from Snapshot: {e}"
                    ))
                })?;
            vfio_common.set_state(state, msi_state, msix_state, mig)?;
            // parse_capabilities() does not run on this path, so capture
            // the same two capability offsets directly - otherwise the
            // D3 and FLR hooks are silently dead on a restored or
            // live-migrated device.
            vfio_common.capture_pcie_and_pm_cap_offsets();
        } else {
            vfio_common.parse_capabilities(bdf)?;
            vfio_common.initialize_legacy_interrupt()?;
        }

        Ok(vfio_common)
    }

    /// In case msix table offset is not page size aligned, we need do some fixup to achieve it.
    /// Because we don't want the MMIO RW region and trap region overlap each other.
    fn fixup_msix_region(&mut self, bar_id: u32, region_size: u64) -> u64 {
        if let Some(msix) = self.interrupt.msix.as_mut() {
            let msix_cap = &mut msix.cap;

            // Suppose table_bir equals to pba_bir here. Am I right?
            let (table_offset, table_size) = msix_cap.table_range();
            if is_page_size_aligned(table_offset) || msix_cap.table_bir() != bar_id {
                return region_size;
            }

            let (pba_offset, pba_size) = msix_cap.pba_range();
            let msix_sz = align_page_size_up(table_size + pba_size);
            // Expand region to hold RW and trap region which both page size aligned
            let size = cmp::max(region_size * 2, msix_sz * 2);
            // let table starts from the middle of the region
            msix_cap.table_set_offset((size / 2) as u32);
            msix_cap.pba_set_offset((size / 2 + pba_offset - table_offset) as u32);

            size
        } else {
            // MSI-X not supported for this device
            region_size
        }
    }

    pub(crate) fn allocate_bars(
        &mut self,
        allocator: &mut SystemAllocator,
        mmio32_allocator: &mut AddressAllocator,
        mmio64_allocator: &mut AddressAllocator,
        resources: Option<&[Resource]>,
    ) -> Result<Vec<PciBarConfiguration>, PciDeviceError> {
        let mut bars = Vec::new();
        let mut bar_id = VFIO_PCI_BAR0_REGION_INDEX;

        // Going through all regular regions to compute the BAR size.
        // We're not saving the BAR address to restore it, because we
        // are going to allocate a guest address for each BAR and write
        // that new address back.
        while bar_id < VFIO_PCI_CONFIG_REGION_INDEX {
            let mut region_size: u64 = 0;
            let mut region_type = PciBarRegionType::Memory32BitRegion;
            let mut prefetchable = PciBarPrefetchable::NotPrefetchable;
            let mut flags: u32 = 0;

            let mut restored_bar_addr = None;
            if let Some(resources) = resources {
                for resource in resources {
                    if let Resource::PciBar {
                        index,
                        base,
                        size,
                        type_,
                        ..
                    } = resource
                        && *index == bar_id as usize
                    {
                        restored_bar_addr = Some(GuestAddress(*base));
                        region_size = *size;
                        region_type = PciBarRegionType::from(*type_);
                        break;
                    }
                }
                if restored_bar_addr.is_none() {
                    bar_id += 1;
                    continue;
                }
            } else {
                let bar_offset = if bar_id == VFIO_PCI_ROM_REGION_INDEX {
                    (PCI_ROM_EXP_BAR_INDEX * 4) as u32
                } else {
                    PCI_CONFIG_BAR_OFFSET + bar_id * 4
                };

                // First read flags
                flags = self.vfio_wrapper.read_config_dword(bar_offset);

                // Is this an IO BAR?
                let io_bar = if bar_id == VFIO_PCI_ROM_REGION_INDEX {
                    false
                } else {
                    matches!(flags & PCI_CONFIG_IO_BAR, PCI_CONFIG_IO_BAR)
                };

                // Is this a 64-bit BAR?
                let is_64bit_bar = if bar_id == VFIO_PCI_ROM_REGION_INDEX {
                    false
                } else {
                    matches!(
                        flags & PCI_CONFIG_MEMORY_BAR_64BIT,
                        PCI_CONFIG_MEMORY_BAR_64BIT
                    )
                };

                if matches!(
                    flags & PCI_CONFIG_BAR_PREFETCHABLE,
                    PCI_CONFIG_BAR_PREFETCHABLE
                ) {
                    prefetchable = PciBarPrefetchable::Prefetchable;
                }

                // To get size write all 1s
                self.vfio_wrapper
                    .write_config_dword(bar_offset, 0xffff_ffff);

                // And read back BAR value. The device will write zeros for bits it doesn't care about
                let mut lower = self.vfio_wrapper.read_config_dword(bar_offset);

                if io_bar {
                    // Mask flag bits (lowest 2 for I/O bars)
                    lower &= !0b11;

                    // BAR is not enabled
                    if lower == 0 {
                        bar_id += 1;
                        continue;
                    }

                    // IO BAR
                    region_type = PciBarRegionType::IoRegion;

                    // Invert bits and add 1 to calculate size
                    region_size = (!lower + 1) as u64;
                } else if is_64bit_bar {
                    // 64 bits Memory BAR
                    region_type = PciBarRegionType::Memory64BitRegion;

                    // Query size of upper BAR of 64-bit BAR
                    let upper_offset: u32 = PCI_CONFIG_BAR_OFFSET + (bar_id + 1) * 4;
                    self.vfio_wrapper
                        .write_config_dword(upper_offset, 0xffff_ffff);
                    let upper = self.vfio_wrapper.read_config_dword(upper_offset);

                    let mut combined_size = (u64::from(upper) << 32) | u64::from(lower);

                    // Mask out flag bits (lowest 4 for memory bars)
                    combined_size &= !0b1111;

                    // BAR is not enabled
                    if combined_size == 0 {
                        bar_id += 1;
                        continue;
                    }

                    // Invert and add 1 to to find size
                    region_size = !combined_size + 1;
                } else {
                    region_type = PciBarRegionType::Memory32BitRegion;

                    // Mask out flag bits (lowest 4 for memory bars)
                    lower &= !0b1111;

                    if lower == 0 {
                        bar_id += 1;
                        continue;
                    }

                    // Invert and add 1 to to find size
                    region_size = (!lower + 1) as u64;
                }
            }

            let bar_addr = match region_type {
                PciBarRegionType::IoRegion => {
                    // The address needs to be 4 bytes aligned.
                    allocator
                        .allocate_io_addresses(restored_bar_addr, region_size, Some(0x4))
                        .ok_or(PciDeviceError::IoAllocationFailed(region_size))?
                }
                PciBarRegionType::Memory32BitRegion => {
                    // BAR allocation must be naturally aligned
                    mmio32_allocator
                        .allocate(restored_bar_addr, region_size, Some(region_size))
                        .ok_or(PciDeviceError::IoAllocationFailed(region_size))?
                }
                PciBarRegionType::Memory64BitRegion => {
                    // We need do some fixup to keep MMIO RW region and msix cap region page size
                    // aligned.
                    region_size = self.fixup_msix_region(bar_id, region_size);
                    mmio64_allocator
                        .allocate(
                            restored_bar_addr,
                            region_size,
                            Some(cmp::max(
                                // SAFETY: FFI call. Trivially safe.
                                unsafe { sysconf(_SC_PAGESIZE) as GuestUsize },
                                region_size,
                            )),
                        )
                        .ok_or(PciDeviceError::IoAllocationFailed(region_size))?
                }
            };

            // We can now build our BAR configuration block.
            let bar = PciBarConfiguration::default()
                .set_index(bar_id as usize)
                .set_address(bar_addr.raw_value())
                .set_size(region_size)
                .set_region_type(region_type)
                .set_prefetchable(prefetchable);

            // Skip on restore as BARs come from the saved PciConfiguration state.
            if resources.is_none() {
                if bar_id == VFIO_PCI_ROM_REGION_INDEX {
                    self.configuration
                        .add_pci_rom_bar(&bar, flags & 0x1)
                        .map_err(|e| {
                            PciDeviceError::IoRegistrationFailed(bar_addr.raw_value(), e)
                        })?;
                } else {
                    self.configuration.add_pci_bar(&bar).map_err(|e| {
                        PciDeviceError::IoRegistrationFailed(bar_addr.raw_value(), e)
                    })?;
                }
            }

            bars.push(bar);
            self.mmio_regions.push(MmioRegion {
                start: bar_addr,
                length: region_size,
                type_: region_type,
                index: bar_id,
                user_memory_regions: Vec::new(),
            });

            bar_id += 1;
            if region_type == PciBarRegionType::Memory64BitRegion {
                bar_id += 1;
            }
        }
        Ok(bars)
    }

    pub(crate) fn free_bars(
        &mut self,
        allocator: &mut SystemAllocator,
        mmio32_allocator: &mut AddressAllocator,
        mmio64_allocator: &mut AddressAllocator,
    ) -> Result<(), PciDeviceError> {
        for region in self.mmio_regions.iter() {
            match region.type_ {
                PciBarRegionType::IoRegion => {
                    allocator.free_io_addresses(region.start, region.length);
                }
                PciBarRegionType::Memory32BitRegion => {
                    mmio32_allocator.free(region.start, region.length);
                }
                PciBarRegionType::Memory64BitRegion => {
                    mmio64_allocator.free(region.start, region.length);
                }
            }
        }
        Ok(())
    }

    fn parse_msix_capabilities(&mut self, cap: u8) -> MsixCap {
        let msg_ctl = self.vfio_wrapper.read_config_word((cap + 2).into());

        let table = self.vfio_wrapper.read_config_dword((cap + 4).into());

        let pba = self.vfio_wrapper.read_config_dword((cap + 8).into());

        MsixCap {
            msg_ctl,
            table,
            pba,
        }
    }

    fn initialize_msix(
        &mut self,
        msix_cap: MsixCap,
        cap_offset: u32,
        bdf: PciBdf,
        state: Option<MsixConfigState>,
    ) {
        let interrupt_source_group = self
            .msi_interrupt_manager
            .create_group(MsiIrqGroupConfig {
                base: 0,
                count: msix_cap.table_size() as InterruptIndex,
            })
            .unwrap();

        let msix_config = MsixConfig::new(
            msix_cap.table_size(),
            MaybeMutInterruptSourceGroup::Immutable(Arc::clone(&interrupt_source_group)),
            bdf.into(),
            state,
        )
        .unwrap();

        self.interrupt.msix = Some(VfioMsix {
            bar: msix_config,
            cap: msix_cap,
            cap_offset,
            interrupt_source_group,
        });
    }

    fn parse_msi_capabilities(&mut self, cap: u8) -> u16 {
        self.vfio_wrapper.read_config_word((cap + 2).into())
    }

    fn initialize_msi(&mut self, msg_ctl: u16, cap_offset: u32, state: Option<MsiConfigState>) {
        let interrupt_source_group = self
            .msi_interrupt_manager
            .create_group(MsiIrqGroupConfig {
                base: 0,
                count: msi_num_enabled_vectors(msg_ctl) as InterruptIndex,
            })
            .unwrap();

        let msi_config =
            MsiConfig::new(msg_ctl, Arc::clone(&interrupt_source_group), state).unwrap();

        self.interrupt.msi = Some(VfioMsi {
            cfg: msi_config,
            cap_offset,
            interrupt_source_group,
        });
    }

    /// Returns true, if the device claims to have a PCI capability list.
    fn has_capabilities(&self) -> bool {
        let status = self.vfio_wrapper.read_config_word(PCI_CONFIG_STATUS_OFFSET);
        status & PCI_CONFIG_STATUS_CAPABILITIES_LIST != 0
    }

    fn get_msix_cap_idx(&self) -> Option<usize> {
        if !self.has_capabilities() {
            return None;
        }

        let mut cap_next = self
            .vfio_wrapper
            .read_config_byte(PCI_CONFIG_CAPABILITY_OFFSET)
            & PCI_CONFIG_CAPABILITY_PTR_MASK;

        while cap_next != 0 {
            let cap_id = self.vfio_wrapper.read_config_byte(cap_next.into());
            if PciCapabilityId::from(cap_id) == PciCapabilityId::MsiX {
                return Some(cap_next as usize);
            }
            let cap_ptr = self.vfio_wrapper.read_config_byte((cap_next + 1).into())
                & PCI_CONFIG_CAPABILITY_PTR_MASK;

            // See parse_capabilities below for an explanation.
            if cap_ptr == cap_next {
                break;
            }
            cap_next = cap_ptr;
        }

        None
    }

    /// Capture the standard-capability-list offsets of the PCI Express
    /// and Power Management capabilities, needed by
    /// `write_config_register`'s D3 and FLR hooks. Deliberately narrow:
    /// unlike `parse_capabilities`, this does not initialize MSI/MSI-X,
    /// add the NVIDIA clique capability, or parse extended capabilities.
    /// It exists so the two offsets are captured on the restore path
    /// too, where `set_state` runs instead of `parse_capabilities` and
    /// neither field is serialized - without this, the D3 and FLR hooks
    /// are silently dead on every snapshot-restored or live-migrated
    /// device.
    fn capture_pcie_and_pm_cap_offsets(&mut self) {
        if !self.has_capabilities() {
            return;
        }

        let mut cap_iter = self
            .vfio_wrapper
            .read_config_byte(PCI_CONFIG_CAPABILITY_OFFSET)
            & PCI_CONFIG_CAPABILITY_PTR_MASK;

        while cap_iter != 0 {
            let cap_id = self.vfio_wrapper.read_config_byte(cap_iter.into());

            match PciCapabilityId::from(cap_id) {
                PciCapabilityId::PciExpress => self.pcie_cap_offset = Some(cap_iter),
                PciCapabilityId::PowerManagement => self.pm_cap_offset = Some(cap_iter),
                _ => {}
            }

            let cap_next = self.vfio_wrapper.read_config_byte((cap_iter + 1).into())
                & PCI_CONFIG_CAPABILITY_PTR_MASK;

            if cap_next == 0 || cap_next == cap_iter {
                break;
            }

            cap_iter = cap_next;
        }
    }

    fn parse_capabilities(&mut self, bdf: PciBdf) -> Result<(), VfioPciError> {
        if !self.has_capabilities() {
            return Ok(());
        }

        self.capture_pcie_and_pm_cap_offsets();

        let mut cap_iter = self
            .vfio_wrapper
            .read_config_byte(PCI_CONFIG_CAPABILITY_OFFSET)
            & PCI_CONFIG_CAPABILITY_PTR_MASK;

        while cap_iter != 0 {
            let cap_id = self.vfio_wrapper.read_config_byte(cap_iter.into());

            match PciCapabilityId::from(cap_id) {
                PciCapabilityId::MessageSignalledInterrupts => {
                    if let Some(irq_info) = self.vfio_wrapper.get_irq_info(VFIO_PCI_MSI_IRQ_INDEX)
                        && irq_info.count > 0
                    {
                        // Parse capability only if the VFIO device
                        // supports MSI.
                        let msg_ctl = self.parse_msi_capabilities(cap_iter);
                        self.initialize_msi(msg_ctl, cap_iter as u32, None);
                    }
                }
                PciCapabilityId::MsiX => {
                    if let Some(irq_info) = self.vfio_wrapper.get_irq_info(VFIO_PCI_MSIX_IRQ_INDEX)
                        && irq_info.count > 0
                    {
                        // Parse capability only if the VFIO device
                        // supports MSI-X.
                        let msix_cap = self.parse_msix_capabilities(cap_iter);
                        self.initialize_msix(msix_cap, cap_iter as u32, bdf, None);
                    }
                }
                // Advertise the device as a PCIe integrated endpoint if the
                // PASID capability is enabled.
                PciCapabilityId::PciExpress
                    if self
                        .extended_caps
                        .iter()
                        .any(|cap| cap.id() == PciExpressCapabilityId::ProcessAddressSpaceId) =>
                {
                    self.present_as_integrated_endpoint(cap_iter);
                }
                _ => {}
            }

            let cap_next = self.vfio_wrapper.read_config_byte((cap_iter + 1).into())
                & PCI_CONFIG_CAPABILITY_PTR_MASK;

            // Break out of the loop, if we either find the end or we have a broken device. This
            // doesn't handle all cases where a device might send us in a loop here, but it
            // handles case of a device returning 0xFF instead of implementing a real
            // capabilities list.
            if cap_next == 0 || cap_next == cap_iter {
                break;
            }

            cap_iter = cap_next;
        }

        if let Some(clique_id) = self.x_nv_gpudirect_clique {
            self.add_nv_gpudirect_clique_cap(cap_iter, clique_id);
        }

        if self.pcie_cap_offset.is_some() && self.pm_cap_offset.is_some() {
            self.parse_extended_capabilities()?;
        }

        Ok(())
    }

    fn patch_reg(&mut self, reg_idx: usize, mask: u32, patch: u32, write_mask: u32) {
        let entry = self.patches.entry(reg_idx).or_insert(ConfigPatch {
            mask: 0,
            patch: 0,
            write_mask: 0,
        });

        entry.mask |= mask;
        entry.patch = (entry.patch & !mask) | (patch & mask);
        entry.write_mask |= write_mask;
    }

    fn add_nv_gpudirect_clique_cap(&mut self, cap_iter: u8, clique_id: u8) {
        // Turing, Ampere, Hopper, and Lovelace GPUs have dedicated space
        // at 0xD4 for this capability.
        let cap_offset = 0xd4u32;

        self.patch_reg((cap_iter / 4) as usize, 0x0000_ff00, cap_offset << 8, 0);

        let reg_idx = (cap_offset / 4) as usize;
        self.patch_reg(reg_idx, 0xffff_ffff, 0x50080009u32, 0);
        self.patch_reg(
            reg_idx + 1,
            0xffff_ffff,
            (u32::from(clique_id) << 19) | 0x5032,
            0,
        );
    }

    fn present_as_integrated_endpoint(&mut self, cap_offset: u8) {
        let reg_idx = (u32::from(cap_offset) / 4) as usize;

        self.patch_reg(reg_idx, PCI_EXP_FLAGS_TYPE_MASK, PCI_EXP_TYPE_RC_END, 0);

        let clear = |this: &mut Self, reg: u32| {
            this.patch_reg(reg_idx + (reg / 4) as usize, 0xffff_ffff, 0, 0);
        };

        clear(self, PCI_EXP_LNKCAP);
        clear(self, PCI_EXP_LNKCTL);

        let flags = self.vfio_wrapper.read_config_dword(u32::from(cap_offset));
        let version = (flags & PCI_EXP_FLAGS_VERS_MASK) >> PCI_EXP_FLAGS_VERS_SHIFT;

        if version > 1 {
            clear(self, PCI_EXP_LNKCAP2);
            clear(self, PCI_EXP_LNKCTL2);
        }
    }

    fn override_next_extended_cap(&mut self, offset: u32, next: u32) {
        self.patch_reg(
            (offset / 4) as usize,
            PCI_EXT_CAP_NEXT_MASK,
            next << PCI_EXT_CAP_NEXT_SHIFT,
            0,
        );
    }

    fn add_extended_cap(
        &mut self,
        last_offset: u32,
        offset: u32,
        occupied: &[u32],
        cap: &dyn PciExpressCapability,
    ) -> Result<(), VfioPciError> {
        let end = offset.checked_add(cap.size());

        if last_offset >= offset
            || !offset.is_multiple_of(PCI_EXT_CAP_ALIGN)
            || offset < PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET
            || end.is_none_or(|end| end > PCIE_CONFIG_SPACE_SIZE)
            || occupied
                .iter()
                .any(|used| (offset..offset + cap.size()).contains(used))
        {
            return Err(VfioPciError::ExtendedCapNoSpace(cap.id() as u16));
        }

        if cap.dwords().len() != cap.write_masks().len() {
            return Err(VfioPciError::ExtendedCapMismatchedWriteMasks(
                cap.id() as u16
            ));
        }

        self.override_next_extended_cap(last_offset, offset);

        let mut reg_idx = (offset / 4) as usize;
        self.patch_reg(
            reg_idx,
            0xffff_ffff,
            (cap.id() as u32) | (cap.version() << 16),
            0,
        );

        for (dword, write_mask) in cap.dwords().iter().zip(cap.write_masks()) {
            reg_idx += 1;
            self.patch_reg(reg_idx, 0xffff_ffff, *dword, *write_mask);
        }

        Ok(())
    }

    fn add_extended_caps(
        &mut self,
        last_offset: u32,
        occupied: &[u32],
    ) -> Result<(), VfioPciError> {
        let caps = self.extended_caps.clone();
        let total: u32 = caps.iter().map(|cap| cap.size()).sum();

        let mut offset = PCIE_CONFIG_SPACE_SIZE.saturating_sub(total);
        let mut last_offset = last_offset;

        for cap in caps {
            self.add_extended_cap(last_offset, offset, occupied, cap.as_ref())?;

            last_offset = offset;
            offset += cap.size();
        }

        Ok(())
    }

    fn parse_extended_capabilities(&mut self) -> Result<(), VfioPciError> {
        let mut current_offset = PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET;
        let mut last_kept_offset: Option<u32> = None;
        let mut occupied = Vec::new();

        loop {
            occupied.push(current_offset);

            let ext_cap_hdr = self.vfio_wrapper.read_config_dword(current_offset);

            let cap_id: u16 = (ext_cap_hdr & 0xffff) as u16;
            let cap_next: u16 = ((ext_cap_hdr >> 20) & 0xfff) as u16;

            match PciExpressCapabilityId::from(cap_id) {
                PciExpressCapabilityId::AlternativeRoutingIdentificationInterpretation
                | PciExpressCapabilityId::ResizeableBar
                | PciExpressCapabilityId::SingleRootIoVirtualization => match last_kept_offset {
                    Some(offset) => self.override_next_extended_cap(offset, cap_next.into()),
                    None => self.patch_reg(
                        (PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET / 4) as usize,
                        0xffff_ffff,
                        (PciExpressCapabilityId::NullCapability as u32)
                            | (u32::from(cap_next) << PCI_EXT_CAP_NEXT_SHIFT),
                        0,
                    ),
                },
                _ => last_kept_offset = Some(current_offset),
            }

            if cap_next == 0 {
                break;
            }

            current_offset = cap_next.into();
        }

        self.add_extended_caps(
            last_kept_offset.unwrap_or(PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET),
            &occupied,
        )
    }

    pub(crate) fn enable_intx(&mut self) -> Result<(), VfioPciError> {
        if let Some(intx) = &mut self.interrupt.intx
            && !intx.enabled
        {
            if let Some(eventfd) = intx.interrupt_source_group.notifier(0) {
                self.vfio_wrapper
                    .enable_irq(VFIO_PCI_INTX_IRQ_INDEX, vec![&eventfd])
                    .map_err(VfioPciError::EnableIntx)?;

                intx.enabled = true;
            } else {
                return Err(VfioPciError::MissingNotifier);
            }
        }

        Ok(())
    }

    pub(crate) fn disable_intx(&mut self) {
        if let Some(intx) = &mut self.interrupt.intx
            && intx.enabled
        {
            if let Err(e) = self.vfio_wrapper.disable_irq(VFIO_PCI_INTX_IRQ_INDEX) {
                error!("Could not disable INTx: {e}");
            } else {
                intx.enabled = false;
            }
        }
    }

    pub(crate) fn enable_msi(&self) -> Result<(), VfioPciError> {
        if let Some(msi) = &self.interrupt.msi {
            let mut irq_fds: Vec<EventFd> = Vec::new();
            for i in 0..msi.cfg.num_enabled_vectors() {
                if let Some(eventfd) = msi.interrupt_source_group.notifier(i as InterruptIndex) {
                    irq_fds.push(eventfd);
                } else {
                    return Err(VfioPciError::MissingNotifier);
                }
            }

            self.vfio_wrapper
                .enable_msi(irq_fds.iter().collect())
                .map_err(VfioPciError::EnableMsi)?;
        }

        Ok(())
    }

    pub(crate) fn disable_msi(&self) {
        if let Err(e) = self.vfio_wrapper.disable_msi() {
            error!("Could not disable MSI: {e}");
        }
    }

    pub(crate) fn enable_msix(&self) -> Result<(), VfioPciError> {
        if let Some(msix) = &self.interrupt.msix {
            let mut irq_fds: Vec<EventFd> = Vec::new();
            for i in 0..msix.bar.table_entries.len() {
                if let Some(eventfd) = msix.interrupt_source_group.notifier(i as InterruptIndex) {
                    irq_fds.push(eventfd);
                } else {
                    return Err(VfioPciError::MissingNotifier);
                }
            }

            self.vfio_wrapper
                .enable_msix(irq_fds.iter().collect())
                .map_err(VfioPciError::EnableMsix)?;
        }

        Ok(())
    }

    pub(crate) fn disable_msix(&self) {
        if let Err(e) = self.vfio_wrapper.disable_msix() {
            error!("Could not disable MSI-X: {e}");
        }
    }

    fn initialize_legacy_interrupt(&mut self) -> Result<(), VfioPciError> {
        if let Some(irq_info) = self.vfio_wrapper.get_irq_info(VFIO_PCI_INTX_IRQ_INDEX)
            && irq_info.count == 0
        {
            // A count of 0 means the INTx IRQ is not supported, therefore
            // it shouldn't be initialized.
            return Ok(());
        }

        if let Some(interrupt_source_group) = self.legacy_interrupt_group.clone() {
            self.interrupt.intx = Some(VfioIntx {
                interrupt_source_group,
                enabled: false,
            });

            self.enable_intx()?;
        }

        Ok(())
    }

    fn update_msi_capabilities(&mut self, offset: u64, data: &[u8]) -> Result<(), VfioPciError> {
        match self.interrupt.update_msi(offset, data) {
            Some(InterruptUpdateAction::EnableMsi) => {
                // Disable INTx before we can enable MSI
                self.disable_intx();
                self.enable_msi()?;
            }
            Some(InterruptUpdateAction::DisableMsi) => {
                // Fallback onto INTx when disabling MSI
                self.disable_msi();
                self.enable_intx()?;
            }
            _ => {}
        }

        Ok(())
    }

    fn update_msix_capabilities(&mut self, offset: u64, data: &[u8]) -> Result<(), VfioPciError> {
        match self.interrupt.update_msix(offset, data) {
            Some(InterruptUpdateAction::EnableMsix) => {
                // Disable INTx before we can enable MSI-X
                self.disable_intx();
                self.enable_msix()?;
            }
            Some(InterruptUpdateAction::DisableMsix) => {
                // Fallback onto INTx when disabling MSI-X
                self.disable_msix();
                self.enable_intx()?;
            }
            _ => {}
        }

        Ok(())
    }

    fn find_region(&self, addr: u64) -> Option<MmioRegion> {
        for region in self.mmio_regions.iter() {
            if addr >= region.start.raw_value()
                && addr < region.start.unchecked_add(region.length).raw_value()
            {
                return Some(region.clone());
            }
        }
        None
    }

    pub(crate) fn read_bar(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        let addr = base + offset;
        if let Some(region) = self.find_region(addr) {
            let offset = addr - region.start.raw_value();

            if self.interrupt.msix_table_accessed(region.index, offset) {
                self.interrupt.msix_read_table(offset, data);
            } else {
                self.vfio_wrapper.region_read(region.index, offset, data);
            }
        }

        // INTx EOI
        // The guest reading from the BAR potentially means the interrupt has
        // been received and can be acknowledged.
        if self.interrupt.intx_in_use()
            && let Err(e) = self.vfio_wrapper.unmask_irq(VFIO_PCI_INTX_IRQ_INDEX)
        {
            error!("Failed unmasking INTx IRQ: {e}");
        }
    }

    pub(crate) fn write_bar(
        &mut self,
        base: u64,
        offset: u64,
        data: &[u8],
    ) -> Option<Arc<Barrier>> {
        let addr = base + offset;
        if let Some(region) = self.find_region(addr) {
            let offset = addr - region.start.raw_value();

            // If the MSI-X table is written to, we need to update our cache.
            if self.interrupt.msix_table_accessed(region.index, offset) {
                self.interrupt.msix_write_table(offset, data);
            } else {
                self.vfio_wrapper.region_write(region.index, offset, data);
            }
        }

        // INTx EOI
        // The guest writing to the BAR potentially means the interrupt has
        // been received and can be acknowledged.
        if self.interrupt.intx_in_use()
            && let Err(e) = self.vfio_wrapper.unmask_irq(VFIO_PCI_INTX_IRQ_INDEX)
        {
            error!("Failed unmasking INTx IRQ: {e}");
        }

        None
    }

    pub(crate) fn write_config_register(
        &mut self,
        reg_idx: usize,
        offset: u64,
        data: &[u8],
    ) -> (Vec<BarReprogrammingParams>, Option<Arc<Barrier>>) {
        // When the guest wants to write to a BAR, we trap it into
        // our local configuration space. We're not reprogramming
        // VFIO device.
        if (PCI_CONFIG_BAR0_INDEX..PCI_CONFIG_BAR0_INDEX + BAR_NUMS).contains(&reg_idx)
            || reg_idx == PCI_ROM_EXP_BAR_INDEX
        {
            // We keep our local cache updated with the BARs.
            // We'll read it back from there when the guest is asking
            // for BARs (see read_config_register()).
            return (
                self.configuration
                    .write_config_register(reg_idx, offset, data),
                None,
            );
        }

        if let Some(patch) = self.patches.get_mut(&reg_idx) {
            patch.write(offset, data);

            if patch.mask == 0xffff_ffff {
                return (Vec::new(), None);
            }
        }

        let reg = (reg_idx * PCI_CONFIG_REGISTER_SIZE) as u64;

        // If the MSI or MSI-X capabilities are accessed, we need to
        // update our local cache accordingly.
        // Depending on how the capabilities are modified, this could
        // trigger a VFIO MSI or MSI-X toggle.
        if let Some((cap_id, cap_base)) = self.interrupt.accessed(reg) {
            let cap_offset: u64 = reg - cap_base + offset;
            match cap_id {
                PciCapabilityId::MessageSignalledInterrupts => {
                    if let Err(e) = self.update_msi_capabilities(cap_offset, data) {
                        error!("Could not update MSI capabilities: {e}");
                    }
                }
                PciCapabilityId::MsiX => {
                    if let Err(e) = self.update_msix_capabilities(cap_offset, data) {
                        error!("Could not update MSI-X capabilities: {e}");
                    }
                }
                _ => {}
            }
        }

        // Make sure to write to the device's PCI config space after MSI/MSI-X
        // interrupts have been enabled/disabled. In case of MSI, when the
        // interrupts are enabled through VFIO (using VFIO_DEVICE_SET_IRQS),
        // the MSI Enable bit in the MSI capability structure found in the PCI
        // config space is disabled by default. That's why when the guest is
        // enabling this bit, we first need to enable the MSI interrupts with
        // VFIO through VFIO_DEVICE_SET_IRQS ioctl, and only after we can write
        // to the device region to update the MSI Enable bit.
        self.vfio_wrapper.write_config((reg + offset) as u32, data);

        // The non BAR write path goes directly to the VFIO device and not the shadow,
        // so the PciConfiguration shadow can get stale. Mirror the write into the
        // shadow here since snapshot() serializes it. Without this the shadow keeps its
        // device init values and a snapshot encodes PCI_COMMAND as zero.
        //
        // Use the raw write_* helpers rather than the PciConfiguration method
        // self.configuration.write_config_register(), which would otherwise drain
        // pending_bar_reprogram, owned by the BAR block below, and rerun MSI-X
        // set_msg_ctl, already done by update_msix_capabilities above.
        let byte_offset = reg_idx * PCI_CONFIG_REGISTER_SIZE + offset as usize;
        match data.len() {
            1 => self.configuration.write_byte(byte_offset, data[0]),
            2 => self
                .configuration
                .write_word(byte_offset, u16::from(data[0]) | (u16::from(data[1]) << 8)),
            4 => self
                .configuration
                .write_reg(reg_idx, LittleEndian::read_u32(data)),
            _ => {}
        }

        // Return pending BAR repgrogramming if MSE bit is set
        let mut ret_param = self.configuration.pending_bar_reprogram();
        if !ret_param.is_empty() {
            if self.read_config_register(COMMAND_REG) & COMMAND_REG_MEMORY_SPACE_MASK
                == COMMAND_REG_MEMORY_SPACE_MASK
            {
                info!("BAR reprogramming parameter is returned: {ret_param:x?}");
                self.configuration.clear_pending_bar_reprogram();
            } else {
                info!(
                    "MSE bit is disabled. No BAR reprogramming parameter is returned: {ret_param:x?}"
                );

                ret_param = Vec::new();
            }
        }

        (ret_param, None)
    }

    pub(crate) fn read_config_register(&mut self, reg_idx: usize) -> u32 {
        // When reading the BARs, we trap it and return what comes
        // from our local configuration space. We want the guest to
        // use that and not the VFIO device BARs as it does not map
        // with the guest address space.
        if (PCI_CONFIG_BAR0_INDEX..PCI_CONFIG_BAR0_INDEX + BAR_NUMS).contains(&reg_idx)
            || reg_idx == PCI_ROM_EXP_BAR_INDEX
        {
            return self.configuration.read_reg(reg_idx);
        }

        if let Some(id) = self.get_msix_cap_idx() {
            let msix = self.interrupt.msix.as_mut().unwrap();
            if reg_idx * 4 == id + 4 {
                return msix.cap.table;
            } else if reg_idx * 4 == id + 8 {
                return msix.cap.pba;
            }
        }

        // Since we don't support passing multi-functions devices, we should
        // mask the multi-function bit, bit 7 of the Header Type byte on the
        // register 3.
        let mask = if reg_idx == PCI_HEADER_TYPE_REG_INDEX {
            0xff7f_ffff
        } else {
            0xffff_ffff
        };

        // The config register read comes from the VFIO device itself.
        let mut value = self.vfio_wrapper.read_config_dword((reg_idx * 4) as u32) & mask;

        if let Some(config_patch) = self.patches.get(&reg_idx) {
            value = (value & !config_patch.mask) | config_patch.patch;
        }

        value
    }

    fn state(&self) -> VfioCommonState {
        let intx_state = self.interrupt.intx.as_ref().map(|intx| IntxState {
            enabled: intx.enabled,
        });

        let msi_state = self.interrupt.msi.as_ref().map(|msi| MsiState {
            cap: msi.cfg.cap,
            cap_offset: msi.cap_offset,
        });

        let msix_state = self.interrupt.msix.as_ref().map(|msix| MsixState {
            cap: msix.cap,
            cap_offset: msix.cap_offset,
            bdf: msix.bar.devid,
        });

        VfioCommonState {
            intx_state,
            msi_state,
            msix_state,
            patches: self.patches.clone(),
        }
    }

    fn set_state(
        &mut self,
        state: &VfioCommonState,
        msi_state: Option<MsiConfigState>,
        msix_state: Option<MsixConfigState>,
        migration_data: Option<VfioMigrationData>,
    ) -> Result<(), VfioPciError> {
        // A snapshot carrying VFIO migration state cannot be restored onto a
        // device without migration support.
        if migration_data.is_some() && self.migration_flags.is_none() {
            return Err(VfioPciError::RestoreMigration(anyhow!(
                "snapshot carries VFIO migration state but the device does not support migration"
            )));
        }

        if let (Some(intx), Some(interrupt_source_group)) =
            (&state.intx_state, self.legacy_interrupt_group.clone())
        {
            self.interrupt.intx = Some(VfioIntx {
                interrupt_source_group,
                enabled: false,
            });

            if intx.enabled {
                self.enable_intx()?;
            }
        }

        if let Some(msi) = &state.msi_state {
            self.initialize_msi(msi.cap.msg_ctl, msi.cap_offset, msi_state);
        }

        if let Some(msix) = &state.msix_state {
            self.initialize_msix(msix.cap, msix.cap_offset, msix.bdf.into(), msix_state);
        }

        // Replay the opaque device state captured at snapshot. The kernel walks
        // the intermediate STOP arc internally, so RESUMING is a single write.
        if let Some(mig) = migration_data {
            let blob = BASE64_STANDARD.decode(mig.blob.as_bytes()).map_err(|e| {
                VfioPciError::RestoreMigration(anyhow!(
                    "Failed to base64-decode migration blob: {e}"
                ))
            })?;
            self.load_migration_data(&blob)
                .context("Failed to load migration data for restoring VFIO device")
                .map_err(VfioPciError::RestoreMigration)?;
        }

        self.patches = state.patches.clone();

        self.sync_command_and_interrupts()?;

        Ok(())
    }

    fn sync_command_and_interrupts(&self) -> Result<(), VfioPciError> {
        // Push PCI_COMMAND to the device. State replay updates the shadow config
        // space but not the device, so memory decode and bus mastering would
        // otherwise stay disabled after restore.
        let cmd = (self.configuration.read_reg(COMMAND_REG) & 0xFFFF) as u16;
        self.vfio_wrapper.write_config(
            (COMMAND_REG * PCI_CONFIG_REGISTER_SIZE) as u32,
            &cmd.to_le_bytes(),
        );

        // Rearm the kernel interrupt eventfds. State replay restores only the
        // MSI or MSI-X state in memory, not the VFIO_DEVICE_SET_IRQS wiring.
        if let Some(msi) = &self.interrupt.msi
            && msi.cfg.enabled()
        {
            self.enable_msi()?;
        } else if let Some(msix) = &self.interrupt.msix
            && msix.bar.enabled()
        {
            self.enable_msix()?;
        }

        Ok(())
    }

    fn transition_migration_state(&self, target: VfioMigrationState) -> anyhow::Result<()> {
        debug!("VFIO migration transition -> {target:?}");
        self.vfio_wrapper
            .set_migration_state(target)
            .with_context(|| format!("VFIO set_migration_state({target:?}) failed"))
    }

    fn reset_and_rearm(&self) {
        self.vfio_wrapper.reset();
        // The reset returns the device to RUNNING. Reapply PCI_COMMAND and the
        // interrupt eventfds the same way the restore path does.
        if let Err(e) = self.sync_command_and_interrupts() {
            error!("VFIO device rearm after reset failed: {e}");
        }
        // Let VfioPciDevice know a reset happened, so it can rebuild any
        // P2P mapping the reset may have dropped. Relaxed: the comparison
        // against this happens later, on the same thread.
        self.reset_generation.fetch_add(1, Ordering::Relaxed);
    }

    // A failed set can leave the device anywhere along the transition path
    // including ERROR, where only a reset recovers. Attempt the recovery
    // state first when the caller names one, otherwise reset directly.
    // The original error is propagated either way.
    pub(crate) fn transition_migration_state_with_recovery(
        &self,
        target: VfioMigrationState,
        recover: Option<VfioMigrationState>,
    ) -> anyhow::Result<()> {
        let Err(err) = self.transition_migration_state(target) else {
            return Ok(());
        };

        if let Some(state) = recover
            && self.transition_migration_state(state).is_ok()
        {
            return Err(err);
        }

        warn!("VFIO device in indeterminate migration state, resetting");
        self.reset_and_rearm();
        Err(err)
    }

    pub(crate) fn save_migration_data(&self) -> Result<Vec<u8>, MigratableError> {
        self.transition_migration_state_with_recovery(
            VfioMigrationState::StopCopy,
            Some(VfioMigrationState::Stop),
        )
        .map_err(MigratableError::Snapshot)?;

        let data = self.vfio_wrapper.read_migration_data();

        // We entered STOP_COPY successfully, so the STOP_COPY to STOP arc is
        // valid whether or not the read succeeded. Return the device to STOP
        // either way and reset it if even that fails.
        let stop = self
            .transition_migration_state_with_recovery(VfioMigrationState::Stop, None)
            .map_err(MigratableError::Snapshot);

        let data = data
            .context("VFIO migration data read failed")
            .map_err(MigratableError::Snapshot)?;
        stop?;
        Ok(data)
    }

    pub(crate) fn load_migration_data(&self, data: &[u8]) -> Result<(), MigratableError> {
        // Leave the device in RESUMING and resume() drives it back to RUNNING.
        self.transition_migration_state_with_recovery(
            VfioMigrationState::Resuming,
            Some(VfioMigrationState::Stop),
        )
        .map_err(MigratableError::Restore)?;

        // A failed write leaves an incomplete RESUMING session, which the uAPI
        // only allows aborting with a reset, so reset directly.
        if let Err(e) = self.vfio_wrapper.write_migration_data(data) {
            self.reset_and_rearm();
            return Err(MigratableError::Restore(
                anyhow::Error::new(e).context("VFIO migration data write failed"),
            ));
        }
        Ok(())
    }

    // No op without ranges to track. page_size is a hint, the device reports
    // back the granularity it actually applied, kept for the report calls.
    pub(crate) fn start_dirty_log(
        &mut self,
        ranges: &[DmaLoggingRange],
        page_size: u64,
    ) -> Result<(), MigratableError> {
        if ranges.is_empty() {
            return Ok(());
        }
        let negotiated = self
            .vfio_wrapper
            .start_dma_logging(page_size, ranges)
            .context("VFIO start_dma_logging failed")
            .map_err(MigratableError::StartDirtyLog)?;
        debug!(
            "VFIO DMA logging started over {} range(s), requested page size {page_size:#x}, device granularity {negotiated:#x}",
            ranges.len()
        );
        self.dma_logging_page_size = Some(negotiated);
        Ok(())
    }

    pub(crate) fn stop_dirty_log(&mut self) -> Result<(), MigratableError> {
        if self.dma_logging_page_size.take().is_none() {
            return Ok(());
        }
        self.vfio_wrapper
            .stop_dma_logging()
            .context("VFIO stop_dma_logging failed")
            .map_err(MigratableError::StopDirtyLog)
    }

    // Reports per range dirty bitmaps from the kernel, merged into a single
    // MemoryRangeTable. Returns an empty table when logging is not active so
    // the caller can splice it into the union without a special case.
    pub(crate) fn dirty_log(
        &self,
        ranges: &[DmaLoggingRange],
    ) -> Result<MemoryRangeTable, MigratableError> {
        let Some(page_size) = self.dma_logging_page_size else {
            return Ok(MemoryRangeTable::default());
        };
        let mut table = MemoryRangeTable::default();
        for range in ranges {
            let range_table = self
                .vfio_wrapper
                .report_dma_logging(*range, page_size)
                .context("VFIO report_dma_logging failed")
                .map_err(MigratableError::DirtyLog)?;
            table.extend(range_table);
        }
        Ok(table)
    }
}

impl Pausable for VfioCommon {}

impl Snapshottable for VfioCommon {
    fn id(&self) -> String {
        String::from(VFIO_COMMON_ID)
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        let mut vfio_common_snapshot = Snapshot::new_from_state(&self.state())?;

        // Snapshot PciConfiguration
        vfio_common_snapshot.add_snapshot(self.configuration.id(), self.configuration.snapshot()?);

        // Snapshot MSI
        if let Some(msi) = &mut self.interrupt.msi {
            vfio_common_snapshot.add_snapshot(msi.cfg.id(), msi.cfg.snapshot()?);
        }

        // Snapshot MSI-X
        if let Some(msix) = &mut self.interrupt.msix {
            vfio_common_snapshot.add_snapshot(msix.bar.id(), msix.bar.snapshot()?);
        }

        if self.migration_flags.is_some() {
            let data = self.save_migration_data()?;
            let mig = VfioMigrationData {
                blob: BASE64_STANDARD.encode(&data),
            };
            vfio_common_snapshot.add_snapshot(
                VFIO_MIGRATION_ID.to_string(),
                Snapshot::new_from_state(&mig)?,
            );
        }

        Ok(vfio_common_snapshot)
    }
}

/// VfioPciDevice represents a VFIO PCI device.
/// This structure implements the BusDevice and PciDevice traits.
///
/// A VfioPciDevice is bound to a VfioDevice and is also a PCI device.
/// The VMM creates a VfioDevice, then assigns it to a VfioPciDevice,
/// which then gets added to the PCI bus.
pub struct VfioPciDevice {
    id: String,
    vm: Arc<dyn hypervisor::Vm>,
    device: Arc<VfioDevice>,
    vfio_ops: Arc<dyn VfioOps>,
    common: VfioCommon,
    iommu_attached: bool,
    // Whether to map VFIO device MMIO BARs into the host IOMMU address space.
    // Required for peer-to-peer DMA between VFIO devices.
    p2p_dma: bool,
    memory_slot_allocator: MemorySlotAllocator,
    // Guest memory layout, used to enumerate the IOVA ranges to track when
    // programming VFIO DMA logging.
    memory: GuestMemoryAtomic<GuestMemoryMmap>,
    bdf: PciBdf,
    device_path: PathBuf,
    // Set once the "kernel/device does not support dma-buf export" warning
    // has fired, so it is logged at most once per device rather than once
    // per BAR (or per BAR per re-map).
    dmabuf_unsupported_warned: bool,
    // The last common.reset_generation value reconcile_after_reset() has
    // rebuilt P2P mappings against. Plain u64, not an atomic: every
    // caller of reconcile_after_reset() already holds &mut self.
    reconciled_reset_generation: u64,
}

/// The dma-buf length for a sparse area, or `None` when the area cannot
/// satisfy the IOAS alignment contract:
///   iova % alignment == 0 && (iova + length) % alignment == 0
/// An alignment of 0 means the backend offers no file-backed path.
fn dma_range_for_area(iova: u64, area_size: u64, iova_alignment: u64) -> Option<u64> {
    if iova_alignment != 0
        && area_size != 0
        && iova.is_multiple_of(iova_alignment)
        && area_size.is_multiple_of(iova_alignment)
    {
        Some(area_size)
    } else {
        None
    }
}

/// How a dma-buf export or import errno maps onto the fallback ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DmaBufOutcome {
    // The kernel or device has no dma-buf export support at all.
    Unsupported,
    // The dma-buf is currently revoked (ENODEV). This only shapes what
    // gets logged, not what happens: at the L1/L2 call sites below it
    // just picks the log wording, and in p2p_restore_all every failure
    // is retried on the next un-revoke edge regardless of
    // classification - Revoked only makes that expected case log at
    // debug! instead of warn!.
    Revoked,
    // A stale IOAS area occupies the IOVA (EEXIST). Also classification
    // for logging only: an EEXIST here falls straight through to the
    // legacy VA mapping (L3) like any other error; nothing unmaps and
    // retries at this call site.
    Stale,
    // A genuine, unexpected failure.
    Failed,
}

/// Classify an errno returned by a dma-buf export or file-backed map
/// attempt onto the fallback ladder above.
fn classify_dma_buf_errno(errno: i32) -> DmaBufOutcome {
    match errno {
        libc::ENOTTY | libc::EINVAL | libc::EOPNOTSUPP => DmaBufOutcome::Unsupported,
        libc::ENODEV => DmaBufOutcome::Revoked,
        libc::EEXIST => DmaBufOutcome::Stale,
        _ => DmaBufOutcome::Failed,
    }
}

/// A short, human-readable gloss for a classified dma-buf layer failure,
/// used to shape the L1/L2 debug log lines in `map_mmio_regions()`. Purely
/// cosmetic at this commit - `Revoked` and `Stale` carry no behavioural
/// consequence here; that arrives with the revoke/rebuild hooks.
fn dma_buf_outcome_gloss(outcome: DmaBufOutcome) -> &'static str {
    match outcome {
        DmaBufOutcome::Unsupported => "this kernel/device has no dma-buf path",
        DmaBufOutcome::Revoked => "the dma-buf is currently revoked",
        DmaBufOutcome::Stale => "a stale IOAS area occupies the IOVA",
        DmaBufOutcome::Failed => "a genuine failure worth attention",
    }
}

/// Map a region's BAR content into the host IOMMU address space: through
/// its dma-buf if it has one, or the legacy VA mapping otherwise. A free
/// function rather than a method on `&self`: a method would need to
/// borrow all of `self`, which conflicts with the `&mut` borrow of the
/// region living inside `self.common.mmio_regions` at the one call site,
/// `move_bar`, that is already live for the duration of the call.
fn p2p_map_region(
    vfio_ops: &dyn VfioOps,
    umr: &mut UserMemoryRegion,
    bar: u32,
) -> Result<(), vfio_ioctls::VfioError> {
    match &umr.dmabuf {
        Some(f) => {
            let ret = vfio_ops.vfio_dma_map_file(umr.start, umr.p2p_len, f.as_raw_fd(), 0);
            if let Err(e) = &ret {
                debug!("Cannot map BAR {bar} through its dma-buf: {e}");
            }
            ret
        }
        None => {
            // vfio_dma_map is unsound and ought to be marked as unsafe
            // SAFETY: MmapRegion invariants guarantee that
            // host_addr points to len bytes of
            // valid memory that will only be unmapped with munmap().
            let ret =
                unsafe { vfio_ops.vfio_dma_map(umr.start, umr.mapping.len(), umr.mapping.addr()) };
            if let Err(e) = &ret {
                debug!("Cannot map BAR {bar} through the legacy VA mapping: {e}");
            }
            ret
        }
    }
}

/// Unmap a region's BAR content from the host IOMMU address space.
/// Always uses `p2p_len`, not `mapping.len()`: a dma-buf-backed mapping
/// may cover less than the guest-visible mmap length. A free function
/// for the same reason as `p2p_map_region` above.
fn p2p_unmap_region(
    vfio_ops: &dyn VfioOps,
    umr: &mut UserMemoryRegion,
) -> Result<(), vfio_ioctls::VfioError> {
    vfio_ops.vfio_dma_unmap(umr.start, umr.p2p_len as usize)
}

/// A memory-space-enable (MSE, COMMAND register bit 1) transition between
/// two reads of the same register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MseEdge {
    Enabled,
    Disabled,
}

/// The memory-space-enable edge between two COMMAND register reads.
/// `None` when MSE is unchanged, whether or not some other bit changed.
fn mse_transition(before: u32, after: u32) -> Option<MseEdge> {
    let before_enabled = before & COMMAND_REG_MEMORY_SPACE_MASK == COMMAND_REG_MEMORY_SPACE_MASK;
    let after_enabled = after & COMMAND_REG_MEMORY_SPACE_MASK == COMMAND_REG_MEMORY_SPACE_MASK;
    match (before_enabled, after_enabled) {
        (true, false) => Some(MseEdge::Disabled),
        (false, true) => Some(MseEdge::Enabled),
        (true, true) | (false, false) => None,
    }
}

// PMCSR (Power Management Control/Status Register) power-state field:
// bits 1:0 of the dword at `pm_cap_offset + 4`. D0 is 0b00.
const PMCSR_POWER_STATE_MASK: u32 = 0b11;
const PMCSR_D0: u32 = 0;

/// A D-state transition between two reads of the PMCSR dword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DStateEdge {
    EnteredLowPower,
    ReturnedToD0,
}

/// The D-state edge between two PMCSR reads. `None` when the power-state
/// field is unchanged, whether or not some other PMCSR bit (e.g. PME_En)
/// changed, and also `None` for a transition between two non-D0 states
/// (D1/D2 are not used by this hook's revoke/restore semantics).
fn d_state_transition(before: u32, after: u32) -> Option<DStateEdge> {
    let before_state = before & PMCSR_POWER_STATE_MASK;
    let after_state = after & PMCSR_POWER_STATE_MASK;
    match (before_state == PMCSR_D0, after_state == PMCSR_D0) {
        (true, false) => Some(DStateEdge::EnteredLowPower),
        (false, true) => Some(DStateEdge::ReturnedToD0),
        (true, true) | (false, false) => None,
    }
}

/// Whether a config-space write to `reg_idx` at byte `offset` (relative
/// to the dword) with the given `data` sets BCR_FLR - bit 15 of the PCIe
/// Device Control register, which occupies the low word of the dword at
/// `pcie_cap_offset + 8`. Detected from the write data itself, not a
/// before/after comparison: vfio-pci performs the reset synchronously
/// inside the forwarded write, so by the time an "after" read would
/// happen the reset has already run.
fn write_sets_flr(pcie_cap_offset: u8, reg_idx: usize, offset: u64, data: &[u8]) -> bool {
    let flr_reg_idx = (pcie_cap_offset as usize + 8) / 4;
    if reg_idx != flr_reg_idx {
        return false;
    }
    // Bit 15 of the dword is bit 7 of byte index 1 (0-based) within it.
    // checked_sub returns None when the write starts after byte index 1,
    // i.e. does not reach it at all.
    const FLR_BYTE_INDEX: usize = 1;
    const FLR_BIT_MASK: u8 = 0x80;
    let write_start = offset as usize;
    let Some(index_in_data) = FLR_BYTE_INDEX.checked_sub(write_start) else {
        return false;
    };
    data.get(index_in_data)
        .is_some_and(|byte| byte & FLR_BIT_MASK != 0)
}

impl VfioPciDevice {
    /// Constructs a new Vfio Pci device for the given Vfio device
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        vm: Arc<dyn hypervisor::Vm>,
        device: Arc<VfioDevice>,
        vfio_ops: Arc<dyn VfioOps>,
        msi_interrupt_manager: Arc<dyn InterruptManager<GroupConfig = MsiIrqGroupConfig>>,
        legacy_interrupt_group: Option<Arc<dyn InterruptSourceGroup>>,
        iommu_attached: bool,
        p2p_dma: bool,
        bdf: PciBdf,
        memory_slot_allocator: MemorySlotAllocator,
        memory: GuestMemoryAtomic<GuestMemoryMmap>,
        snapshot: Option<&Snapshot>,
        x_nv_gpudirect_clique: Option<u8>,
        x_exclude_mmap_bars: Vec<u8>,
        device_path: PathBuf,
        extended_caps: Vec<Arc<dyn PciExpressCapability + Send + Sync>>,
    ) -> Result<Self, VfioPciError> {
        let vfio_wrapper = VfioDeviceWrapper::new(Arc::clone(&device));

        let common = VfioCommon::new(
            msi_interrupt_manager,
            legacy_interrupt_group,
            Arc::new(vfio_wrapper) as Arc<dyn Vfio>,
            &PciVfioSubclass::VfioSubclass,
            bdf,
            vm_migration::snapshot_from_id(snapshot, VFIO_COMMON_ID),
            VfioCommonConfig {
                x_nv_gpudirect_clique,
                x_exclude_mmap_bars,
                extended_caps,
            },
        )?;

        let vfio_pci_device = VfioPciDevice {
            id,
            vm,
            device,
            vfio_ops,
            common,
            iommu_attached,
            p2p_dma,
            memory_slot_allocator,
            memory,
            bdf,
            device_path,
            dmabuf_unsupported_warned: false,
            reconciled_reset_generation: 0,
        };

        Ok(vfio_pci_device)
    }

    pub fn iommu_attached(&self) -> bool {
        self.iommu_attached
    }

    fn generate_sparse_areas(
        caps: &[VfioRegionInfoCap],
        region_index: u32,
        region_start: u64,
        region_size: u64,
        vfio_msix: Option<&VfioMsix>,
    ) -> Result<Vec<VfioRegionSparseMmapArea>, VfioPciError> {
        for cap in caps {
            match cap {
                VfioRegionInfoCap::SparseMmap(sparse_mmap) => return Ok(sparse_mmap.areas.clone()),
                VfioRegionInfoCap::MsixMappable => {
                    if !is_4k_aligned(region_start) {
                        error!(
                            "Region start address 0x{region_start:x} must be at least aligned on 4KiB"
                        );
                        return Err(VfioPciError::RegionAlignment);
                    }
                    if !is_4k_multiple(region_size) {
                        error!("Region size 0x{region_size:x} must be at least a multiple of 4KiB");
                        return Err(VfioPciError::RegionSize);
                    }

                    // In case the region contains the MSI-X vectors table or
                    // the MSI-X PBA table, we must calculate the subregions
                    // around them, leading to a list of sparse areas.
                    // We want to make sure we will still trap MMIO accesses
                    // to these MSI-X specific ranges. If these region don't align
                    // with pagesize, we can achieve it by enlarging its range.
                    //
                    // Using a BtreeMap as the list provided through the iterator is sorted
                    // by key. This ensures proper split of the whole region.
                    let mut inter_ranges = BTreeMap::new();
                    if let Some(msix) = vfio_msix {
                        if region_index == msix.cap.table_bir() {
                            let (offset, size) = msix.cap.table_range();
                            let offset = align_page_size_down(offset);
                            let size = align_page_size_up(size);
                            // MSI-X mmap region safety: when a device has a non page
                            // aligned MSI-X offset, fixup_msix_region() relocates MSI-X
                            // to the upper half of an enlarged virtual BAR, causing the
                            // offsets in msix.cap to exceed the physical BAR size. This
                            // check skips carving a hole, preventing invalid offsets from
                            // reaching the mmap path. With no holes,
                            // generate_sparse_areas() returns a single sparse region
                            // covering the entire physical BAR. The relocated MSI-X in
                            // the virtual BAR remains trapped because its upper half has
                            // no mmap backing. Exposing the physical MSI-X region through
                            // mmap is safe when the kernel advertises
                            // VFIO_REGION_INFO_CAP_MSIX_MAPPABLE. When MSI-X offsets are
                            // already page aligned, fixup_msix_region() does not relocate
                            // and this check is satisfied, so a hole is carved at the
                            // intended offset as before.
                            if offset < region_size {
                                inter_ranges.insert(offset, size);
                            }
                        }
                        if region_index == msix.cap.pba_bir() {
                            let (offset, size) = msix.cap.pba_range();
                            let offset = align_page_size_down(offset);
                            let size = align_page_size_up(size);
                            // See MSI-X mmap safety comment above.
                            if offset < region_size {
                                inter_ranges.insert(offset, size);
                            }
                        }
                    }

                    let mut sparse_areas = Vec::new();
                    let mut current_offset = 0;
                    for (range_offset, range_size) in inter_ranges {
                        if range_offset > current_offset {
                            sparse_areas.push(VfioRegionSparseMmapArea {
                                offset: current_offset,
                                size: range_offset - current_offset,
                            });
                        }
                        current_offset = align_page_size_down(range_offset + range_size);
                    }

                    if region_size > current_offset {
                        sparse_areas.push(VfioRegionSparseMmapArea {
                            offset: current_offset,
                            size: region_size - current_offset,
                        });
                    }

                    return Ok(sparse_areas);
                }
                _ => {}
            }
        }

        // In case no relevant capabilities have been found, create a single
        // sparse area corresponding to the entire MMIO region.
        Ok(vec![VfioRegionSparseMmapArea {
            offset: 0,
            size: region_size,
        }])
    }

    /// Whether `map_mmio_regions()` must fail closed for this region
    /// under `x_nv_gpudirect_clique`, and why. `None` means keep
    /// today's silent behavior for this region (map it, skip it
    /// quietly, or let a later check decide) - the caller interprets
    /// `None` exactly as it did before this function existed. All of
    /// the clique fail-closed decision lives here: the ROM/I/O-BAR
    /// exemption, the four skip reasons, and generate_sparse_areas()'s
    /// FIRST-cap-wins semantics for telling a genuine kernel-reported
    /// sparse hole apart from a deliberate MSI-X table/PBA carve-out.
    ///
    /// Each check below short-circuits before touching the parameters
    /// only a later check needs, so a call site that has not yet
    /// computed those later facts may pass a placeholder (`false`,
    /// `0`, `&[]`, or `(0, 0)`) for them without affecting the result.
    #[allow(clippy::too_many_arguments)]
    fn clique_skip_reason(
        clique_configured: bool,
        region_index: u32,
        region_type: PciBarRegionType,
        region_flags: u32,
        caps: &[VfioRegionInfoCap],
        user_excluded: bool,
        is_msix_table_or_pba: bool,
        mapped_vs_size: (u64, u64),
    ) -> Option<CliqueSkipReason> {
        if !clique_configured
            || region_index == VFIO_PCI_ROM_REGION_INDEX
            || region_type == PciBarRegionType::IoRegion
        {
            return None;
        }

        if user_excluded {
            return Some(CliqueSkipReason::UserExcludedBar);
        }

        if region_flags & VFIO_REGION_INFO_FLAG_MMAP == 0 {
            return Some(CliqueSkipReason::NotMmapCapable);
        }

        if is_msix_table_or_pba && !caps.contains(&VfioRegionInfoCap::MsixMappable) {
            return Some(CliqueSkipReason::MsixNotMappable);
        }

        // Mirror generate_sparse_areas()'s FIRST-cap-wins semantics: a
        // region can carry both SparseMmap and MsixMappable, and
        // generate_sparse_areas() takes whichever comes first in the
        // kernel-reported list. Only when SparseMmap is that first
        // match is this a genuine kernel-reported hole; when
        // MsixMappable wins instead, the "hole" is the MSI-X
        // table/PBA carve-out, deliberately trapped by design, not a
        // missing mapping.
        let first_relevant_cap = caps.iter().find(|cap| {
            matches!(
                cap,
                VfioRegionInfoCap::SparseMmap(_) | VfioRegionInfoCap::MsixMappable
            )
        });
        let is_kernel_reported_sparse =
            matches!(first_relevant_cap, Some(VfioRegionInfoCap::SparseMmap(_)));
        if is_kernel_reported_sparse {
            let (mapped, size) = mapped_vs_size;
            if mapped < size {
                return Some(CliqueSkipReason::KernelSparseHole { mapped, size });
            }
        }

        None
    }

    /// Map MMIO regions into the guest, and avoid VM exits when the guest tries
    /// to reach those regions.
    ///
    /// # Arguments
    ///
    /// * `vm` - The VM object. It is used to set the VFIO MMIO regions
    ///   as user memory regions.
    /// * `mem_slot` - The closure to return a memory slot.
    pub fn map_mmio_regions(&mut self) -> Result<(), VfioPciError> {
        let fd = self.device.as_raw_fd();
        // SAFETY: fd is guaranteed valid
        let fd = unsafe { BorrowedFd::borrow_raw(fd) };
        // Query the alignment IOMMU_IOAS_MAP_FILE requires of an IOVA range,
        // once per call rather than once per region or per area. unwrap_or(0)
        // together with dma_range_for_area() returning None for alignment 0
        // is what makes a backend without the file-backed path (the legacy
        // container backend, an mshv-only build, or a kernel too old for
        // IOMMU_IOAS_IOVA_RANGES) fall through cleanly to the legacy mapping
        // below.
        let iova_alignment = self.vfio_ops.vfio_dma_iova_alignment().unwrap_or(0);
        for region in self.common.mmio_regions.iter_mut() {
            let clique_configured = self.common.x_nv_gpudirect_clique.is_some();

            if self
                .common
                .x_exclude_mmap_bars
                .contains(&(region.index as u8))
            {
                if let Some(reason) = Self::clique_skip_reason(
                    clique_configured,
                    region.index,
                    region.type_,
                    0,
                    &[],
                    true,
                    false,
                    (0, 0),
                ) {
                    return Err(VfioPciError::CliqueMappingSkipped(
                        self.bdf,
                        region.index,
                        reason,
                    ));
                }
                info!(
                    "Skipping VFIO BAR mmap and P2P DMA mapping for device {} at {} BAR {} (size = 0x{:x})",
                    self.bdf,
                    self.device_path.display(),
                    region.index,
                    region.length
                );
                continue;
            }

            let region_flags = self.device.get_region_flags(region.index);
            if region_flags & VFIO_REGION_INFO_FLAG_MMAP != 0 {
                let mut prot = 0;
                if region_flags & VFIO_REGION_INFO_FLAG_READ != 0 {
                    prot |= libc::PROT_READ;
                }
                if region_flags & VFIO_REGION_INFO_FLAG_WRITE != 0 {
                    prot |= libc::PROT_WRITE;
                }

                // Retrieve the list of capabilities found on the region
                let caps = if region_flags & VFIO_REGION_INFO_FLAG_CAPS != 0 {
                    self.device.get_region_caps(region.index)
                } else {
                    Vec::new()
                };

                // Don't try to mmap the region if it contains MSI-X table or
                // MSI-X PBA subregion, and if we couldn't find MSIX_MAPPABLE
                // in the list of supported capabilities.
                let is_msix_table_or_pba =
                    self.common.interrupt.msix.as_ref().is_some_and(|msix| {
                        region.index == msix.cap.table_bir() || region.index == msix.cap.pba_bir()
                    });
                if is_msix_table_or_pba && !caps.contains(&VfioRegionInfoCap::MsixMappable) {
                    if let Some(reason) = Self::clique_skip_reason(
                        clique_configured,
                        region.index,
                        region.type_,
                        region_flags,
                        &caps,
                        false,
                        true,
                        (0, 0),
                    ) {
                        return Err(VfioPciError::CliqueMappingSkipped(
                            self.bdf,
                            region.index,
                            reason,
                        ));
                    }
                    continue;
                }

                let mmap_size = self.device.get_region_size(region.index);
                let mmap_offset = self.device.get_region_offset(region.index);

                let sparse_areas = Self::generate_sparse_areas(
                    &caps,
                    region.index,
                    region.start.0,
                    mmap_size,
                    self.common.interrupt.msix.as_ref(),
                )?;

                let mapped_size: u64 = sparse_areas.iter().map(|area| area.size).sum();
                if let Some(reason) = Self::clique_skip_reason(
                    clique_configured,
                    region.index,
                    region.type_,
                    region_flags,
                    &caps,
                    false,
                    false,
                    (mapped_size, mmap_size),
                ) {
                    return Err(VfioPciError::CliqueMappingSkipped(
                        self.bdf,
                        region.index,
                        reason,
                    ));
                }

                let page_size = get_page_size();
                for area in sparse_areas.iter() {
                    // KVM_SET_USER_MEMORY_REGION requires memory_size to be a
                    // multiple of the host page size. On aarch64 with 64K pages
                    // a device BAR can be smaller than a page (e.g. 16K NVMe
                    // BAR).
                    //
                    // The kernel only sets VFIO_REGION_INFO_FLAG_MMAP on sub-page
                    // BARs after verifying the physical BAR start is page-aligned
                    // and reserving the rest of the page. Expansion is only safe
                    // at offset 0 where the kernel reservation applies.
                    //
                    // fixup_msix_region() ensures MSI-X relocation at >= page_size
                    // offset, so the expanded mmap cannot overlap the trap region.
                    let mmap_len = if area.size < page_size {
                        if area.offset != 0 {
                            error!(
                                "BAR {}: sub-page sparse area at non-zero offset 0x{:x} \
                                 cannot be safely expanded to page size",
                                region.index, area.offset,
                            );
                            return Err(VfioPciError::MmapArea);
                        }
                        info!(
                            "BAR {}: expanding sub-page sparse area mmap from 0x{:x} to \
                             page size 0x{:x}",
                            region.index, area.size, page_size,
                        );
                        page_size
                    } else {
                        area.size
                    };
                    let mapping = match MmapRegion::mmap(
                        mmap_len,
                        prot,
                        fd,
                        mmap_offset,
                        area.offset,
                    ) {
                        Ok(mapping) => mapping,
                        Err(_) => {
                            error!(
                                "Could not mmap sparse area (offset = 0x{:x}, size = 0x{:x}): {}",
                                mmap_offset,
                                mmap_len,
                                io::Error::last_os_error()
                            );
                            return Err(VfioPciError::MmapArea);
                        }
                    };

                    let mut user_memory_region = UserMemoryRegion {
                        slot: self.memory_slot_allocator.next_memory_slot(),
                        start: region.start.0 + area.offset,
                        mapping: Arc::new(mapping),
                        p2p_mapped: false,
                        p2p_len: 0,
                        dmabuf: None,
                        p2p_revoked: false,
                    };
                    // SAFETY: MmapRegion invariants guarantee that
                    // user_memory_region.mapping.addr() points to
                    // user_memory_region.mapping.len() bytes of
                    // valid memory that will only be unmapped with munmap().
                    unsafe {
                        self.vm.create_user_memory_region(
                            user_memory_region.slot,
                            user_memory_region.start,
                            user_memory_region.mapping.len(),
                            user_memory_region.mapping.addr(),
                            false,
                            false,
                            hypervisor::MemoryVisibility::Shared,
                        )
                    }
                    .map_err(VfioPciError::CreateUserMemoryRegion)?;

                    // Map the MMIO BAR into the host IOMMU address space.
                    // Only needed if p2p_dma is enabled, and best effort:
                    // some kernels cannot map MMIO (PFNMAP) memory into
                    // their IOMMU backend, while the guest-visible mapping
                    // above already succeeded, so losing this map only
                    // disables DMA from other devices into this BAR. Tries
                    // a dma-buf-backed mapping first (L1/L2), because that
                    // is what can later be re-established from a vCPU
                    // thread; falls back to the legacy VA mapping (L3),
                    // which is today's working path on this host and must
                    // stay reachable on any kernel that lacks dma-buf
                    // export support.
                    if !self.iommu_attached && self.p2p_dma {
                        let mut dmabuf_layer_err: Option<String> = None;

                        // L1 export: try to hand this area to the kernel as
                        // a dma-buf. Exports area.size, never mmap_len - the
                        // page-expanded length would exceed
                        // pci_resource_len and could describe a range past
                        // the BAR.
                        let mapped_via_dmabuf = match dma_range_for_area(
                            user_memory_region.start,
                            area.size,
                            iova_alignment,
                        ) {
                            Some(len) => {
                                match self.device.export_dma_buf(region.index, area.offset, len) {
                                    Ok(Some(file)) => {
                                        // L2 map: the dma-buf's own offset
                                        // space begins at 0 - the range was
                                        // already selected at export time
                                        // via area.offset - so the start
                                        // argument here is always 0.
                                        match self.vfio_ops.vfio_dma_map_file(
                                            user_memory_region.start,
                                            len,
                                            file.as_raw_fd(),
                                            0,
                                        ) {
                                            Ok(()) => {
                                                debug!(
                                                    "Mapped BAR {} of device {} at {} into the \
                                                     host IOMMU address space through a dma-buf \
                                                     (iova 0x{:x}, size 0x{len:x}).",
                                                    region.index,
                                                    self.bdf,
                                                    self.device_path.display(),
                                                    user_memory_region.start,
                                                );
                                                user_memory_region.p2p_mapped = true;
                                                user_memory_region.p2p_len = len;
                                                user_memory_region.dmabuf = Some(Arc::new(file));
                                                true
                                            }
                                            Err(e) => {
                                                // Classify by errno where the
                                                // error carries one, to shape
                                                // the log line; the outcome
                                                // has no behavioural effect
                                                // at this commit (that
                                                // arrives with the
                                                // revoke/rebuild hooks) - it
                                                // only changes what gets
                                                // logged.
                                                match e.errno().map(classify_dma_buf_errno) {
                                                    Some(outcome) => debug!(
                                                        "Cannot map dma-buf for BAR {} of device \
                                                         {} at {} into the host IOMMU address \
                                                         space (iova 0x{:x}, size 0x{len:x}, \
                                                         {}): {e}. Falling back to the legacy VA \
                                                         mapping.",
                                                        region.index,
                                                        self.bdf,
                                                        self.device_path.display(),
                                                        user_memory_region.start,
                                                        dma_buf_outcome_gloss(outcome),
                                                    ),
                                                    None => debug!(
                                                        "Cannot map dma-buf for BAR {} of device \
                                                         {} at {} into the host IOMMU address \
                                                         space (iova 0x{:x}, size 0x{len:x}): \
                                                         {e}. Falling back to the legacy VA \
                                                         mapping.",
                                                        region.index,
                                                        self.bdf,
                                                        self.device_path.display(),
                                                        user_memory_region.start,
                                                    ),
                                                }
                                                dmabuf_layer_err = Some(e.to_string());
                                                false
                                            }
                                        }
                                    }
                                    Ok(None) => {
                                        if !self.dmabuf_unsupported_warned {
                                            // info!, not warn!: on a kernel
                                            // without dma-buf export support
                                            // (e.g. this project's own golden
                                            // kernel 316) this is the
                                            // expected, non-degraded path,
                                            // not a problem worth a warning
                                            // on every boot.
                                            info!(
                                                "Device {} at {} does not support exporting BAR \
                                                 MMIO as a dma-buf; peer-to-peer DMA will use \
                                                 the legacy VA mapping.",
                                                self.bdf,
                                                self.device_path.display(),
                                            );
                                            self.dmabuf_unsupported_warned = true;
                                        }
                                        // L1 was attempted and reported
                                        // unsupported, not skipped - distinct
                                        // from the None arm below, where the
                                        // alignment predicate skipped it
                                        // before any attempt.
                                        dmabuf_layer_err = Some("unsupported".to_string());
                                        false
                                    }
                                    Err(e) => {
                                        // Classify by errno where the error
                                        // carries one, to shape the log
                                        // line; the outcome has no
                                        // behavioural effect at this commit
                                        // (that arrives with the
                                        // revoke/rebuild hooks) - it only
                                        // changes what gets logged.
                                        match e.errno().map(classify_dma_buf_errno) {
                                            Some(outcome) => debug!(
                                                "Cannot export BAR {} of device {} at {} as a \
                                                 dma-buf ({}): {e}. Falling back to the legacy \
                                                 VA mapping.",
                                                region.index,
                                                self.bdf,
                                                self.device_path.display(),
                                                dma_buf_outcome_gloss(outcome),
                                            ),
                                            None => debug!(
                                                "Cannot export BAR {} of device {} at {} as a \
                                                 dma-buf: {e}. Falling back to the legacy VA \
                                                 mapping.",
                                                region.index,
                                                self.bdf,
                                                self.device_path.display(),
                                            ),
                                        }
                                        dmabuf_layer_err = Some(e.to_string());
                                        false
                                    }
                                }
                            }
                            None => false,
                        };

                        // L3 legacy: reached from Ok(None), from any L1/L2
                        // error, or when L1 was skipped (area could not
                        // satisfy the alignment contract). Unchanged from
                        // before this series.
                        if !mapped_via_dmabuf {
                            // vfio_dma_map should be unsafe but isn't.
                            // SAFETY: MmapRegion invariants guarantee that
                            // user_memory_region.mapping.addr() points to
                            // user_memory_region.mapping.len() bytes of
                            // valid memory that will only be unmapped with munmap().
                            match unsafe {
                                self.vfio_ops.vfio_dma_map(
                                    user_memory_region.start,
                                    user_memory_region.mapping.len(),
                                    user_memory_region.mapping.addr(),
                                )
                            } {
                                Ok(()) => {
                                    user_memory_region.p2p_mapped = true;
                                    user_memory_region.p2p_len =
                                        user_memory_region.mapping.len() as u64;
                                }
                                Err(e) if self.common.x_nv_gpudirect_clique.is_some() => {
                                    // L4: x_nv_gpudirect_clique asserts
                                    // peer-to-peer DMA is required, so its
                                    // loss stays fatal here and only here -
                                    // the flag asserts the capability, not
                                    // the mechanism, and making it fatal at
                                    // L1/L2 would turn every clique user on
                                    // a pre-6.19 kernel - including this
                                    // project's own GH200 on kernel 316 -
                                    // from booting-and-working into
                                    // refusing-to-boot. Push the region
                                    // first: the KVM slot exists and must
                                    // be reclaimed on teardown.
                                    region.user_memory_regions.push(user_memory_region);
                                    return Err(VfioPciError::P2pDmaMapAllLayersFailed {
                                        source: e,
                                        path: self.device_path.clone(),
                                        bdf: self.bdf,
                                        bar: region.index,
                                        dmabuf: dmabuf_layer_err
                                            .unwrap_or_else(|| "not attempted".to_string()),
                                    });
                                }
                                Err(e) => warn!(
                                    "Cannot map BAR {} of device {} at {} into the host \
                                     IOMMU address space (iova 0x{:x}, size 0x{:x}): {e}. \
                                     Guest access is unaffected; peer-to-peer DMA into \
                                     this BAR is disabled.",
                                    region.index,
                                    self.bdf,
                                    self.device_path.display(),
                                    user_memory_region.start,
                                    user_memory_region.mapping.len(),
                                ),
                            }
                        }
                    }
                    region.user_memory_regions.push(user_memory_region);
                }
            } else if let Some(reason) = Self::clique_skip_reason(
                clique_configured,
                region.index,
                region.type_,
                region_flags,
                &[],
                false,
                false,
                (0, 0),
            ) {
                return Err(VfioPciError::CliqueMappingSkipped(
                    self.bdf,
                    region.index,
                    reason,
                ));
            }
        }

        for region in self.common.mmio_regions.iter() {
            let extents: Vec<String> = region
                .user_memory_regions
                .iter()
                .map(|umr| {
                    format!(
                        "0x{:x}..0x{:x}",
                        umr.start,
                        umr.start + umr.mapping.len() as u64
                    )
                })
                .collect();
            info!(
                "Device {} BAR {}: mapped guest-PA extents [{}]",
                self.bdf,
                region.index,
                extents.join(", ")
            );
        }

        Ok(())
    }

    pub fn unmap_mmio_regions(&mut self) {
        for region in self.common.mmio_regions.iter_mut() {
            for user_memory_region in region.user_memory_regions.drain(..) {
                let len = user_memory_region.mapping.len();
                let host_addr = user_memory_region.mapping.addr();
                // Unmap MMIO region from the host IOMMU address space via VfioOps
                // Only for regions that were actually P2P-mapped. Uses p2p_len,
                // not len: the two are always equal today, but a dma-buf-backed
                // mapping (added later) may cover less than the guest-visible
                // mmap length.
                let p2p_len = user_memory_region.p2p_len as usize;
                if user_memory_region.p2p_mapped
                    && let Err(e) = self
                        .vfio_ops
                        .vfio_dma_unmap(user_memory_region.start, p2p_len)
                        .map_err(|e| VfioPciError::DmaUnmap(e, self.device_path.clone(), self.bdf))
                {
                    error!(
                        "Could not unmap MMIO region from the host IOMMU address space: \
                            iova 0x{:x}, size 0x{:x}: {}, ",
                        user_memory_region.start, p2p_len, e
                    );
                }

                // Remove region
                // SAFETY: only valid entries are added to the user_memory_regions field
                // of the entries of self.common.mmio_regions.
                // Also, host_addr..host_addr + len is valid by the MmapRegion invariants.
                if let Err(e) = unsafe {
                    self.vm.remove_user_memory_region(
                        user_memory_region.slot,
                        user_memory_region.start,
                        len,
                        host_addr,
                        false,
                    )
                } {
                    error!("Could not remove the userspace memory region: {e}");
                }

                self.memory_slot_allocator
                    .free_memory_slot(user_memory_region.slot);
            }
        }
    }

    /// Drop the host IOMMU mapping for every dma-buf-backed region that is
    /// currently mapped, in response to a guest edge that revokes the
    /// dma-buf (memory-space-enable disabled, D3, or FLR). Regions with
    /// `dmabuf == None` are untouched: the kernel's revoke is a dma-buf
    /// mechanism, a legacy PFNMAP mapping is never revoked, and leaving
    /// those regions alone is what keeps this hook inert on a kernel
    /// without dma-buf export support. Never fatal: this runs on a vCPU
    /// thread in response to a guest config-space write.
    fn p2p_revoke_all(&mut self) {
        for region in self.common.mmio_regions.iter_mut() {
            for umr in region.user_memory_regions.iter_mut() {
                if umr.dmabuf.is_none() || !umr.p2p_mapped {
                    continue;
                }
                // IOMMU_IOAS_UNMAP is permitted while revoked (measured: T5).
                if let Err(e) = self
                    .vfio_ops
                    .vfio_dma_unmap(umr.start, umr.p2p_len as usize)
                {
                    warn!(
                        "Could not unmap the revoked dma-buf for BAR {} of device {} at {} \
                         (iova 0x{:x}, size 0x{:x}): {e}",
                        region.index,
                        self.bdf,
                        self.device_path.display(),
                        umr.start,
                        umr.p2p_len,
                    );
                }
                // Whether or not the unmap above succeeded: the kernel's
                // revoke already dropped the mapping on its side, and a
                // stale entry here would only cause a spurious unmap
                // later.
                umr.p2p_mapped = false;
                umr.p2p_revoked = true;
                debug!(
                    "BAR {} of device {} at {} dropped the host IOMMU mapping \
                     for a revoked dma-buf (iova 0x{:x}, size 0x{:x}).",
                    region.index,
                    self.bdf,
                    self.device_path.display(),
                    umr.start,
                    umr.p2p_len,
                );
            }
        }
    }

    /// Re-establish the host IOMMU mapping for every region a revoke edge
    /// dropped, in response to the matching un-revoke guest edge. Regions
    /// with `dmabuf == None` are untouched, for the same reason as
    /// `p2p_revoke_all`. Never fatal, for the same reason too.
    fn p2p_restore_all(&mut self) {
        for region in self.common.mmio_regions.iter_mut() {
            for umr in region.user_memory_regions.iter_mut() {
                if !umr.p2p_revoked {
                    continue;
                }
                let Some(fd) = umr.dmabuf.as_ref().map(|f| f.as_raw_fd()) else {
                    continue;
                };
                // UNMAP-then-MAP_FILE, always, and unconditionally of
                // whether there is anything to unmap: a stale iopt_area
                // provably survives a full revoke/un-revoke cycle and
                // answers EEXIST on MAP_FILE otherwise (measured: T12).
                let _ = self
                    .vfio_ops
                    .vfio_dma_unmap(umr.start, umr.p2p_len as usize);
                match self
                    .vfio_ops
                    .vfio_dma_map_file(umr.start, umr.p2p_len, fd, 0)
                {
                    Ok(()) => {
                        umr.p2p_mapped = true;
                        umr.p2p_revoked = false;
                        debug!(
                            "BAR {} of device {} at {} re-established the host IOMMU \
                             mapping through its dma-buf (iova 0x{:x}, size 0x{:x}).",
                            region.index,
                            self.bdf,
                            self.device_path.display(),
                            umr.start,
                            umr.p2p_len,
                        );
                    }
                    Err(e)
                        if e.errno().map(classify_dma_buf_errno)
                            == Some(DmaBufOutcome::Revoked) =>
                    {
                        // Expected, not a failure: retried on the next
                        // un-revoke edge.
                        debug!(
                            "BAR {} of device {} at {} is still revoked, will retry on the \
                             next un-revoke edge: {e}",
                            region.index,
                            self.bdf,
                            self.device_path.display(),
                        );
                    }
                    Err(e) => {
                        warn!(
                            "Could not restore the dma-buf mapping for BAR {} of device {} at \
                             {} (iova 0x{:x}, size 0x{:x}): {e}",
                            region.index,
                            self.bdf,
                            self.device_path.display(),
                            umr.start,
                            umr.p2p_len,
                        );
                    }
                }
            }
        }
    }

    /// Rebuild P2P mappings after a reset that `VfioCommon::reset_and_rearm`
    /// performed since the last call, detected via the reset generation
    /// counter. Called at exactly the four `VfioPciDevice`-level entry
    /// points that can reach `reset_and_rearm`: `Pausable::pause`,
    /// `Pausable::resume`, `Snapshottable::snapshot`, and
    /// `Migratable::start_migration`. The restore path is deliberately
    /// excluded: it runs during construction, before `map_mmio_regions`,
    /// when no P2P mapping exists yet.
    fn reconcile_after_reset(&mut self) {
        let generation = self.common.reset_generation.load(Ordering::Relaxed);
        if generation != self.reconciled_reset_generation {
            self.reconciled_reset_generation = generation;
            self.p2p_revoke_all();
            self.p2p_restore_all();
        }
    }

    pub fn mmio_regions(&self) -> Vec<MmioRegion> {
        self.common.mmio_regions.clone()
    }

    // IOVA ranges for DMA logging. Without a virtual IOMMU the device sees an
    // identity mapping of guest memory (iova == gpa), so these are the guest
    // memory regions. A virtual IOMMU is refused in start_migration, see
    // issue #8567.
    fn dirty_log_iova_ranges(&self) -> Vec<DmaLoggingRange> {
        let mem = self.memory.memory();
        mem.iter()
            .map(|region| DmaLoggingRange {
                iova: region.start_addr().raw_value(),
                length: region.len(),
            })
            .collect()
    }
}

impl Drop for VfioPciDevice {
    fn drop(&mut self) {
        self.unmap_mmio_regions();

        if let Some(msix) = &self.common.interrupt.msix
            && msix.bar.enabled()
        {
            self.common.disable_msix();
        }

        if let Some(msi) = &self.common.interrupt.msi
            && msi.cfg.enabled()
        {
            self.common.disable_msi();
        }

        if self.common.interrupt.intx_in_use() {
            self.common.disable_intx();
        }
    }
}

impl BusDevice for VfioPciDevice {
    fn read(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        self.read_bar(base, offset, data);
    }

    fn write(&mut self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        self.write_bar(base, offset, data)
    }
}

// Offset of the 16-bit status register in the PCI configuration space.
const PCI_CONFIG_STATUS_OFFSET: u32 = 0x06;
// Status bit indicating the presence of a capabilities list.
const PCI_CONFIG_STATUS_CAPABILITIES_LIST: u16 = 1 << 4;
// First BAR offset in the PCI config space.
const PCI_CONFIG_BAR_OFFSET: u32 = 0x10;
// Capability register offset in the PCI config space.
const PCI_CONFIG_CAPABILITY_OFFSET: u32 = 0x34;
// The valid bits for the capabilities pointer.
const PCI_CONFIG_CAPABILITY_PTR_MASK: u8 = !0b11;
// Extended capabilities register offset in the PCI config space.
const PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET: u32 = 0x100;
// IO BAR when first BAR bit is 1.
const PCI_CONFIG_IO_BAR: u32 = 0x1;
// 64-bit memory bar flag.
const PCI_CONFIG_MEMORY_BAR_64BIT: u32 = 0x4;
// Prefetchable BAR bit
const PCI_CONFIG_BAR_PREFETCHABLE: u32 = 0x8;
// PCI config register size (4 bytes).
const PCI_CONFIG_REGISTER_SIZE: usize = 4;
// Number of BARs for a PCI device
const BAR_NUMS: usize = 6;
// PCI Header Type register index
const PCI_HEADER_TYPE_REG_INDEX: usize = 3;
// First BAR register index
const PCI_CONFIG_BAR0_INDEX: usize = 4;
// PCI ROM expansion BAR register index
const PCI_ROM_EXP_BAR_INDEX: usize = 12;

impl PciDevice for VfioPciDevice {
    fn allocate_bars(
        &mut self,
        allocator: &mut SystemAllocator,
        mmio32_allocator: &mut AddressAllocator,
        mmio64_allocator: &mut AddressAllocator,
        resources: Option<Vec<Resource>>,
    ) -> Result<Vec<PciBarConfiguration>, PciDeviceError> {
        self.common.allocate_bars(
            allocator,
            mmio32_allocator,
            mmio64_allocator,
            resources.as_deref(),
        )
    }

    fn free_bars(
        &mut self,
        allocator: &mut SystemAllocator,
        mmio32_allocator: &mut AddressAllocator,
        mmio64_allocator: &mut AddressAllocator,
    ) -> Result<(), PciDeviceError> {
        self.common
            .free_bars(allocator, mmio32_allocator, mmio64_allocator)
    }

    fn write_config_register(
        &mut self,
        reg_idx: usize,
        offset: u64,
        data: &[u8],
    ) -> (Vec<BarReprogrammingParams>, Option<Arc<Barrier>>) {
        let p2p_watched = !self.iommu_attached && self.p2p_dma;

        // FLR hook (R2=A): detected from the write data itself, before
        // forwarding, rather than from a before/after COMMAND-register
        // comparison like MSE and D3 below - vfio-pci performs the reset
        // synchronously inside the forwarded write, wrapped in
        // pci_save_state/pci_restore_state, and it is unverified whether
        // MSE is observably cleared around that.
        let sets_flr = p2p_watched
            && self
                .common
                .pcie_cap_offset
                .is_some_and(|off| write_sets_flr(off, reg_idx, offset, data));
        if sets_flr {
            self.p2p_revoke_all();
        }

        // Reads pass straight through to the device, so sample the real
        // register on both sides of the forwarded write rather than
        // trusting the shadow. reg_idx, not a hardcoded COMMAND_REG,
        // because this same before/after pattern also covers the PMCSR
        // register for the D3 hook below.
        let watch_mse = p2p_watched && reg_idx == COMMAND_REG;
        let watch_pmcsr = p2p_watched
            && self
                .common
                .pm_cap_offset
                .is_some_and(|off| reg_idx == (off as usize + 4) / 4);
        let watch = watch_mse || watch_pmcsr;
        let before = if watch {
            self.common.read_config_register(reg_idx)
        } else {
            0
        };
        let ret = self.common.write_config_register(reg_idx, offset, data);
        if watch {
            let after = self.common.read_config_register(reg_idx);
            if watch_mse {
                match mse_transition(before, after) {
                    Some(MseEdge::Disabled) => self.p2p_revoke_all(),
                    Some(MseEdge::Enabled) => self.p2p_restore_all(),
                    None => {}
                }
            } else {
                match d_state_transition(before, after) {
                    Some(DStateEdge::EnteredLowPower) => self.p2p_revoke_all(),
                    Some(DStateEdge::ReturnedToD0) => self.p2p_restore_all(),
                    None => {}
                }
            }
        }

        if sets_flr {
            self.p2p_restore_all();
        }

        ret
    }

    fn read_config_register(&mut self, reg_idx: usize) -> u32 {
        self.common.read_config_register(reg_idx)
    }

    fn read_bar(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        self.common.read_bar(base, offset, data);
    }

    fn write_bar(&mut self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        self.common.write_bar(base, offset, data)
    }

    fn move_bar(&mut self, old_base: u64, new_base: u64) -> Result<(), io::Error> {
        for region in self.common.mmio_regions.iter_mut() {
            if region.start.raw_value() == old_base {
                region.start = GuestAddress(new_base);

                for user_memory_region in region.user_memory_regions.iter_mut() {
                    let len = user_memory_region.mapping.len();
                    let host_addr = user_memory_region.mapping.addr();
                    // Unmap the old MMIO region from the host IOMMU address space.
                    // Only for regions that were actually P2P-mapped. p2p_unmap_region
                    // uses p2p_len, not len: the two are always equal today, but a
                    // dma-buf-backed mapping may cover less than the guest-visible
                    // mmap length.
                    let p2p_len = user_memory_region.p2p_len;
                    if user_memory_region.p2p_mapped
                        && let Err(e) = p2p_unmap_region(self.vfio_ops.as_ref(), user_memory_region)
                            .map_err(|e| {
                                VfioPciError::DmaUnmap(e, self.device_path.clone(), self.bdf)
                            })
                    {
                        error!(
                            "Could not unmap MMIO region from the host IOMMU address space: \
iova 0x{:x}, size 0x{:x}: {}, ",
                            user_memory_region.start, p2p_len, e
                        );
                    }
                    // Remove old region
                    // SAFETY: MmapRegion invariants guarantee that
                    // host_addr points to len bytes of
                    // valid memory that will only be unmapped with munmap().
                    unsafe {
                        self.vm.remove_user_memory_region(
                            user_memory_region.slot,
                            user_memory_region.start,
                            len,
                            host_addr,
                            false,
                        )
                    }
                    .map_err(io::Error::other)?;

                    // Update the user memory region with the correct start address.
                    if new_base > old_base {
                        user_memory_region.start += new_base - old_base;
                    } else {
                        user_memory_region.start -= old_base - new_base;
                    }

                    // Insert new region
                    // SAFETY: MmapRegion invariants guarantee that
                    // host_addr points to len bytes of
                    // valid memory that will only be unmapped with munmap().
                    unsafe {
                        self.vm.create_user_memory_region(
                            user_memory_region.slot,
                            user_memory_region.start,
                            len,
                            host_addr,
                            false,
                            false,
                            hypervisor::MemoryVisibility::Shared,
                        )
                    }
                    .map_err(io::Error::other)?;

                    // Map the moved MMIO region into the host IOMMU address
                    // space, through its dma-buf if it has one or the legacy
                    // VA mapping otherwise. Only regions that were P2P-mapped
                    // before the move; best effort like the initial mapping,
                    // so a refusal cannot fail the guest's BAR reprogramming
                    // after the KVM slot has already moved. A lost mapping is
                    // not retried on later moves: dmabuf is left cached (the
                    // export is still valid) and the revoke flag introduced
                    // later is deliberately not set here, so a move failure
                    // is never mistaken for a revoke.
                    if user_memory_region.p2p_mapped
                        && let Err(e) =
                            p2p_map_region(self.vfio_ops.as_ref(), user_memory_region, region.index)
                    {
                        error!(
                            "Cannot re-map moved BAR {} of device {} at {} into \
                             the host IOMMU address space (iova 0x{:x}, size \
                             0x{:x}): {e}. Peer-to-peer DMA into this BAR is \
                             disabled.",
                            region.index,
                            self.bdf,
                            self.device_path.display(),
                            user_memory_region.start,
                            len,
                        );
                        user_memory_region.p2p_mapped = false;
                    }
                }
            }
        }

        Ok(())
    }

    fn restore_bar_addr(&mut self, params: &BarReprogrammingParams) {
        self.common.configuration.restore_bar_addr(params);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn id(&self) -> Option<String> {
        Some(self.id.clone())
    }
}

impl Pausable for VfioPciDevice {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        let ret = if self.common.migration_flags.is_some() {
            self.common
                .transition_migration_state_with_recovery(VfioMigrationState::Stop, None)
                .map_err(MigratableError::Pause)
        } else {
            Ok(())
        };
        // Reconcile regardless of the transition's own outcome: a failed
        // transition is exactly the case that can fall back to a reset.
        self.reconcile_after_reset();
        ret
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        let ret = if self.common.migration_flags.is_some() {
            self.common
                .transition_migration_state_with_recovery(VfioMigrationState::Running, None)
                .map_err(MigratableError::Resume)
        } else {
            Ok(())
        };
        self.reconcile_after_reset();
        ret
    }
}

impl Snapshottable for VfioPciDevice {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        // Snapshot VfioCommon. Reconcile before propagating any error:
        // save_migration_data's own recovery path can reset the device.
        let common_snapshot = self.common.snapshot();
        self.reconcile_after_reset();

        let mut vfio_pci_dev_snapshot = Snapshot::default();
        vfio_pci_dev_snapshot.add_snapshot(self.common.id(), common_snapshot?);
        Ok(vfio_pci_dev_snapshot)
    }
}

impl Transportable for VfioPciDevice {}

impl Migratable for VfioPciDevice {
    fn notify_started_migration(&mut self) -> result::Result<(), MigratableError> {
        // Reject a device that does not implement migration v2 up front,
        // rather than silently skipping its state and dirty tracking.
        let ret = if self.common.migration_flags.is_none() {
            Err(MigratableError::MigrateSend(anyhow!(
                "VFIO device does not support migration"
            )))
        } else if self.iommu_attached {
            // Dirty tracking behind a virtual IOMMU needs IOVA to GPA
            // translation and would have to follow mappings the guest
            // changes mid migration, neither of which is implemented,
            // see issue #8567.
            Err(MigratableError::MigrateSend(anyhow!(
                "VFIO device live migration is not supported behind a virtual IOMMU"
            )))
        } else {
            Ok(())
        };
        // Reconcile even though this function does not itself trigger a
        // reset: it is the natural checkpoint before migration begins,
        // and catching up on any reconciliation left pending by an
        // earlier pause()/resume()/snapshot() call is cheap (a no-op
        // when the reset generation has not moved) and never wrong.
        self.reconcile_after_reset();
        ret
    }

    fn start_dirty_log(&mut self) -> result::Result<(), MigratableError> {
        let ranges = self.dirty_log_iova_ranges();
        self.common.start_dirty_log(&ranges, get_page_size())
    }

    fn stop_dirty_log(&mut self) -> result::Result<(), MigratableError> {
        self.common.stop_dirty_log()
    }

    fn dirty_log(&mut self) -> result::Result<MemoryRangeTable, MigratableError> {
        let ranges = self.dirty_log_iova_ranges();
        self.common.dirty_log(&ranges)
    }
}

/// This structure implements the ExternalDmaMapping trait. It is meant to
/// be used when the caller tries to provide a way to update the mappings
/// associated with a specific VfioOps instance.
pub struct VfioDmaMapping<M: GuestAddressSpace> {
    vfio_ops: Arc<dyn VfioOps>,
    memory: Arc<M>,
    mmio_regions: Arc<Mutex<Vec<MmioRegion>>>,
}

impl<M: GuestAddressSpace> VfioDmaMapping<M> {
    /// Create a DmaMapping object.
    /// # Parameters
    /// * `vfio_ops`: VfioOps instance.
    /// * `memory`: guest memory to mmap.
    /// * `mmio_regions`: mmio_regions to mmap.
    pub fn new(
        vfio_ops: Arc<dyn VfioOps>,
        memory: Arc<M>,
        mmio_regions: Arc<Mutex<Vec<MmioRegion>>>,
    ) -> Self {
        VfioDmaMapping {
            vfio_ops,
            memory,
            mmio_regions,
        }
    }
}

impl<M: GuestAddressSpace + Sync + Send> ExternalDmaMapping for VfioDmaMapping<M>
where
    M::M: GuestMemoryBackend,
{
    fn map(&self, iova: u64, gpa: u64, size: u64) -> result::Result<(), io::Error> {
        let Ok(usize_size): Result<usize, _> = size.try_into() else {
            return Err(io::Error::other(format!("size {size} overflows usize")));
        };
        let mem = self.memory.memory();
        let guest_addr = GuestAddress(gpa);
        let user_addr = if mem.check_range(guest_addr, usize_size) {
            match mem.get_slice(guest_addr, usize_size) {
                Ok(t) => {
                    assert!(t.len() >= usize_size);
                    Ok(t.ptr_guard_mut())
                }
                Err(e) => {
                    return Err(io::Error::other(format!(
                        "unable to retrieve user address for gpa 0x{gpa:x} from guest memory region: {e}"
                    )));
                }
            }
        } else if self.mmio_regions.lock().unwrap().check_range(gpa, size) {
            Err(self
                .mmio_regions
                .lock()
                .unwrap()
                .find_user_address(gpa, size)?)
        } else {
            return Err(io::Error::other(format!(
                "failed to locate guest address 0x{gpa:x} in guest memory"
            )));
        };
        let user_addr = match user_addr {
            Ok(p) => p.as_ptr(),
            Err(p) => p,
        };

        // vfio_dma_map is unsound and ought to be marked as unsafe
        // SAFETY: find_user_address and GuestMemoryBackend::get_slice() guarantee that
        // the returned pointer is valid for up to `usize_size` bytes.
        // `usize_size` is always equal to `size` due to the above `try_into()` call.
        unsafe { self.vfio_ops.vfio_dma_map(iova, size as usize, user_addr) }.map_err(|e| {
            io::Error::other(format!(
                "failed to map memory into the host IOMMU address space, \
                         iova 0x{iova:x}, gpa 0x{gpa:x}, size 0x{size:x}: {e:?}"
            ))
        })
    }

    fn unmap(&self, iova: u64, size: u64) -> result::Result<(), io::Error> {
        self.vfio_ops
            .vfio_dma_unmap(iova, size as usize)
            .map_err(|e| {
                io::Error::other(format!(
                    "failed to unmap memory from the host IOMMU address space, \
                     iova 0x{iova:x}, size 0x{size:x}: {e:?}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::os::fd::AsFd;
    use std::sync::Mutex;

    use vfio_ioctls::VfioRegionInfoCapSparseMmap;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::PasidCap;

    fn sparse_mmio_regions() -> Vec<MmioRegion> {
        let page_size = get_page_size();
        let file =
            TempFile::new_with_prefix(temp_dir().join("cloud-hypervisor-vfio-sparse-")).unwrap();
        file.as_file().set_len(2 * page_size).unwrap();

        let mapping_a = Arc::new(
            MmapRegion::mmap(
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                file.as_file().as_fd(),
                0,
                0,
            )
            .unwrap(),
        );
        let mapping_b = Arc::new(
            MmapRegion::mmap(
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                file.as_file().as_fd(),
                page_size,
                0,
            )
            .unwrap(),
        );

        vec![MmioRegion {
            start: GuestAddress(page_size),
            length: 4 * page_size,
            type_: PciBarRegionType::Memory32BitRegion,
            index: 0,
            user_memory_regions: vec![
                UserMemoryRegion {
                    slot: 0,
                    start: page_size,
                    p2p_len: mapping_a.len() as u64,
                    mapping: mapping_a,
                    p2p_mapped: false,
                    dmabuf: None,
                    p2p_revoked: false,
                },
                UserMemoryRegion {
                    slot: 1,
                    start: 3 * page_size,
                    p2p_len: mapping_b.len() as u64,
                    mapping: mapping_b,
                    p2p_mapped: false,
                    dmabuf: None,
                    p2p_revoked: false,
                },
            ],
        }]
    }

    #[test]
    fn dma_range_must_fit_single_mmap_area() {
        let page_size = get_page_size();
        let regions = sparse_mmio_regions();
        let guest_addr = page_size + page_size / 2;

        // The BAR spans [P, 5P), but the mmap-backed areas [P, 2P) and [3P, 4P)
        // are non-contiguous. [1.5P, 2.5P) fits the BAR but crosses the unmapped gap.
        let pointer = regions
            .find_user_address(guest_addr, page_size / 2)
            .unwrap();
        assert!(!pointer.is_null());

        assert!(regions.check_range(guest_addr, page_size));
        regions
            .find_user_address(guest_addr, page_size)
            .unwrap_err();
    }

    // Trait default behavior and state enum round trip.

    #[test]
    fn vfio_migration_state_round_trips() {
        for state in [
            VfioMigrationState::Error,
            VfioMigrationState::Stop,
            VfioMigrationState::Running,
            VfioMigrationState::StopCopy,
            VfioMigrationState::Resuming,
            VfioMigrationState::RunningP2P,
            VfioMigrationState::PreCopy,
            VfioMigrationState::PreCopyP2P,
        ] {
            let raw = u32::from(state);
            assert_eq!(VfioMigrationState::try_from(raw).unwrap(), state);
        }
    }

    #[test]
    fn vfio_migration_state_invalid_errors() {
        match VfioMigrationState::try_from(999_u32) {
            Err(VfioError::InvalidMigrationState(999)) => {}
            other => panic!("expected InvalidMigrationState(999), got {other:?}"),
        }
    }

    struct DefaultVfio;
    impl Vfio for DefaultVfio {}

    #[test]
    fn default_migration_flags_returns_none() {
        assert!(matches!(DefaultVfio.migration_flags(), Ok(None)));
    }

    #[test]
    fn default_set_migration_state_errors() {
        assert!(matches!(
            DefaultVfio.set_migration_state(VfioMigrationState::Stop),
            Err(VfioError::NoMigrationSupport)
        ));
    }

    #[test]
    fn default_dma_logging_methods_error() {
        let range = DmaLoggingRange {
            iova: 0,
            length: 0x1000,
        };
        assert!(matches!(
            DefaultVfio.start_dma_logging(0x1000, &[range]),
            Err(VfioError::NoMigrationSupport)
        ));
        assert!(matches!(
            DefaultVfio.stop_dma_logging(),
            Err(VfioError::NoMigrationSupport)
        ));
        assert!(matches!(
            DefaultVfio.report_dma_logging(range, 0x1000),
            Err(VfioError::NoMigrationSupport)
        ));
    }

    // Save and load state machine flows, driven through a mock Vfio wrapper
    // that records state transitions and stores the migration data in memory.

    #[derive(Default)]
    struct MockVfioState {
        transitions: Vec<VfioMigrationState>,
        save_blob: Vec<u8>,
        loaded: Vec<u8>,
        fail_at: Vec<VfioMigrationState>,
        fail_read: bool,
        fail_write: bool,
        resets: u32,
        dma_logging_started: bool,
        dma_logging_page_size: u64,
        dma_logging_ranges: Vec<DmaLoggingRange>,
        dma_logging_bitmap: Vec<u64>,
        dma_logging_negotiated: Option<u64>,
    }

    struct MockVfio {
        state: Mutex<MockVfioState>,
    }

    impl MockVfio {
        fn with_state(state: MockVfioState) -> Arc<Self> {
            Arc::new(Self {
                state: Mutex::new(state),
            })
        }

        fn for_save(blob: Vec<u8>) -> Arc<Self> {
            Self::with_state(MockVfioState {
                save_blob: blob,
                ..Default::default()
            })
        }

        fn for_load() -> Arc<Self> {
            Self::with_state(MockVfioState::default())
        }

        fn failing_at(targets: &[VfioMigrationState]) -> Arc<Self> {
            Self::with_state(MockVfioState {
                fail_at: targets.to_vec(),
                ..Default::default()
            })
        }

        fn failing_read() -> Arc<Self> {
            Self::with_state(MockVfioState {
                fail_read: true,
                ..Default::default()
            })
        }

        fn for_dma_logging(bitmap: Vec<u64>) -> Arc<Self> {
            Self::with_state(MockVfioState {
                dma_logging_bitmap: bitmap,
                ..Default::default()
            })
        }

        fn transitions(&self) -> Vec<VfioMigrationState> {
            self.state.lock().unwrap().transitions.clone()
        }

        fn loaded(&self) -> Vec<u8> {
            self.state.lock().unwrap().loaded.clone()
        }

        fn resets(&self) -> u32 {
            self.state.lock().unwrap().resets
        }

        fn dma_logging_started(&self) -> bool {
            self.state.lock().unwrap().dma_logging_started
        }

        fn dma_logging_recorded(&self) -> (u64, Vec<DmaLoggingRange>) {
            let s = self.state.lock().unwrap();
            (s.dma_logging_page_size, s.dma_logging_ranges.clone())
        }
    }

    impl Vfio for MockVfio {
        fn set_migration_state(&self, state: VfioMigrationState) -> Result<(), VfioError> {
            let mut s = self.state.lock().unwrap();
            s.transitions.push(state);
            if s.fail_at.contains(&state) {
                return Err(VfioError::NoMigrationSupport);
            }
            Ok(())
        }

        fn read_migration_data(&self) -> Result<Vec<u8>, VfioError> {
            let s = self.state.lock().unwrap();
            if s.fail_read {
                return Err(VfioError::NoMigrationSupport);
            }
            Ok(s.save_blob.clone())
        }

        fn write_migration_data(&self, data: &[u8]) -> Result<(), VfioError> {
            let mut s = self.state.lock().unwrap();
            if s.fail_write {
                return Err(VfioError::NoMigrationSupport);
            }
            s.loaded.extend_from_slice(data);
            Ok(())
        }

        fn reset(&self) {
            self.state.lock().unwrap().resets += 1;
        }

        fn start_dma_logging(
            &self,
            page_size: u64,
            ranges: &[DmaLoggingRange],
        ) -> Result<u64, VfioError> {
            let mut s = self.state.lock().unwrap();
            s.dma_logging_started = true;
            s.dma_logging_page_size = page_size;
            s.dma_logging_ranges = ranges.to_vec();
            Ok(s.dma_logging_negotiated.unwrap_or(page_size))
        }

        fn stop_dma_logging(&self) -> Result<(), VfioError> {
            self.state.lock().unwrap().dma_logging_started = false;
            Ok(())
        }

        fn report_dma_logging(
            &self,
            range: DmaLoggingRange,
            page_size: u64,
        ) -> Result<MemoryRangeTable, VfioError> {
            let bitmap = self.state.lock().unwrap().dma_logging_bitmap.clone();
            Ok(MemoryRangeTable::from_dirty_bitmap(
                bitmap, range.iova, page_size,
            ))
        }

        fn region_write(&self, _index: u32, _offset: u64, _data: &[u8]) {}
    }

    struct MockMsiInterruptManager;
    impl InterruptManager for MockMsiInterruptManager {
        type GroupConfig = MsiIrqGroupConfig;
        fn create_group(&self, _: MsiIrqGroupConfig) -> io::Result<Arc<dyn InterruptSourceGroup>> {
            unimplemented!("not exercised by the migration-helper tests")
        }
        fn destroy_group(&self, _: Arc<dyn InterruptSourceGroup>) -> io::Result<()> {
            Ok(())
        }
    }

    fn test_vfio_common<V: Vfio + 'static>(
        vfio_wrapper: Arc<V>,
        migration_flags: Option<u64>,
    ) -> VfioCommon {
        let configuration = PciConfiguration::new(
            0,
            0,
            0,
            PciClassCode::Other,
            &PciVfioSubclass::VfioSubclass,
            None,
            PciHeaderType::Device,
            0,
            0,
            None,
            None,
        );
        VfioCommon {
            configuration,
            mmio_regions: Vec::new(),
            interrupt: Interrupt {
                intx: None,
                msi: None,
                msix: None,
            },
            msi_interrupt_manager: Arc::new(MockMsiInterruptManager),
            legacy_interrupt_group: None,
            vfio_wrapper,
            patches: HashMap::new(),
            x_nv_gpudirect_clique: None,
            x_exclude_mmap_bars: Vec::new(),
            migration_flags,
            dma_logging_page_size: None,
            extended_caps: Vec::new(),
            pcie_cap_offset: None,
            pm_cap_offset: None,
            reset_generation: AtomicU64::new(0),
        }
    }

    #[test]
    fn save_migration_data_success_path() {
        let blob = b"hello migration".to_vec();
        let mock = MockVfio::for_save(blob.clone());
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let got = common.save_migration_data().unwrap();
        assert_eq!(got, blob);
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::StopCopy, VfioMigrationState::Stop]
        );
    }

    #[test]
    fn save_migration_data_stop_copy_failure_recovers_to_stop() {
        let mock = MockVfio::failing_at(&[VfioMigrationState::StopCopy]);
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.save_migration_data().unwrap_err();
        assert!(matches!(err, MigratableError::Snapshot(_)));
        // Entering STOP_COPY failed, so STOP is attempted as recovery.
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::StopCopy, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 0);
    }

    #[test]
    fn save_migration_data_stop_copy_and_stop_failure_resets() {
        let mock = MockVfio::failing_at(&[VfioMigrationState::StopCopy, VfioMigrationState::Stop]);
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.save_migration_data().unwrap_err();
        assert!(matches!(err, MigratableError::Snapshot(_)));
        // The recovery STOP failed too, so the device is reset.
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::StopCopy, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 1);
    }

    #[test]
    fn save_migration_data_read_failure_returns_to_stop() {
        let mock = MockVfio::failing_read();
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.save_migration_data().unwrap_err();
        assert!(matches!(err, MigratableError::Snapshot(_)));
        // STOP_COPY was entered, so the device is returned to STOP.
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::StopCopy, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 0);
    }

    #[test]
    fn save_migration_data_read_and_stop_failure_resets() {
        let mock = MockVfio::with_state(MockVfioState {
            fail_read: true,
            fail_at: vec![VfioMigrationState::Stop],
            ..Default::default()
        });
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.save_migration_data().unwrap_err();
        assert!(matches!(err, MigratableError::Snapshot(_)));
        // The return to STOP after the failed read has no recovery state,
        // so its failure resets the device directly.
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::StopCopy, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 1);
    }

    #[test]
    fn load_migration_data_success_path() {
        let blob = b"restore me".to_vec();
        let mock = MockVfio::for_load();
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        common.load_migration_data(&blob).unwrap();
        assert_eq!(mock.loaded(), blob);
        // Device is left in RESUMING so resume() can drive it to RUNNING.
        assert_eq!(mock.transitions(), vec![VfioMigrationState::Resuming]);
    }

    #[test]
    fn load_migration_data_recovers_on_failure() {
        let mock = MockVfio::failing_at(&[VfioMigrationState::Resuming]);
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.load_migration_data(b"ignored").unwrap_err();
        assert!(matches!(err, MigratableError::Restore(_)));
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::Resuming, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 0);
    }

    #[test]
    fn load_migration_data_resuming_and_stop_failure_resets() {
        let mock = MockVfio::failing_at(&[VfioMigrationState::Resuming, VfioMigrationState::Stop]);
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.load_migration_data(b"ignored").unwrap_err();
        assert!(matches!(err, MigratableError::Restore(_)));
        assert_eq!(
            mock.transitions(),
            vec![VfioMigrationState::Resuming, VfioMigrationState::Stop]
        );
        assert_eq!(mock.resets(), 1);
    }

    #[test]
    fn load_migration_data_write_failure_resets() {
        let mock = MockVfio::with_state(MockVfioState {
            fail_write: true,
            ..Default::default()
        });
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common.load_migration_data(b"ignored").unwrap_err();
        assert!(matches!(err, MigratableError::Restore(_)));
        // A RESUMING session can only be aborted by a reset, so a failed write
        // resets directly rather than attempting STOP.
        assert_eq!(mock.transitions(), vec![VfioMigrationState::Resuming]);
        assert_eq!(mock.resets(), 1);
    }

    #[test]
    fn mock_dma_logging_start_records_and_negotiates() {
        let mock = MockVfio::with_state(MockVfioState {
            dma_logging_negotiated: Some(0x2000),
            ..Default::default()
        });
        let ranges = [
            DmaLoggingRange {
                iova: 0,
                length: 0x1000,
            },
            DmaLoggingRange {
                iova: 0x4000,
                length: 0x2000,
            },
        ];
        // The device may apply a different granularity than requested, so
        // the negotiated page size flows back through the trait.
        let negotiated = mock.start_dma_logging(0x1000, &ranges).unwrap();
        assert_eq!(negotiated, 0x2000);
        assert!(mock.dma_logging_started());
        let (page_size, recorded) = mock.dma_logging_recorded();
        assert_eq!(page_size, 0x1000);
        assert_eq!(recorded, ranges);
    }

    #[test]
    fn mock_dma_logging_stop_clears_started_flag() {
        let mock = MockVfio::for_dma_logging(Vec::new());
        let range = DmaLoggingRange {
            iova: 0,
            length: 0x1000,
        };
        mock.start_dma_logging(0x1000, &[range]).unwrap();
        assert!(mock.dma_logging_started());
        mock.stop_dma_logging().unwrap();
        assert!(!mock.dma_logging_started());
    }

    #[test]
    fn mock_dma_logging_report_converts_bitmap_to_ranges() {
        // Bits 0, 1, 4 set means three dirty pages with a one page gap.
        // Expect two ranges, [iova .. iova+2*ps) and [iova+4*ps .. iova+5*ps).
        let bitmap = vec![0b10011_u64];
        let mock = MockVfio::for_dma_logging(bitmap);
        let page_size: u64 = 0x1000;
        let range = DmaLoggingRange {
            iova: 0x1_0000_0000,
            length: page_size * 64,
        };
        let table = mock.report_dma_logging(range, page_size).unwrap();
        let ranges = table.ranges();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].gpa, 0x1_0000_0000);
        assert_eq!(ranges[0].length, page_size * 2);
        assert_eq!(ranges[1].gpa, 0x1_0000_0000 + 4 * page_size);
        assert_eq!(ranges[1].length, page_size);
    }

    #[test]
    fn start_dirty_log_skips_empty_ranges() {
        let mock = MockVfio::for_dma_logging(Vec::new());
        let mut common = test_vfio_common(Arc::clone(&mock), Some(1));

        // No ranges to track, so no logging session is opened and the
        // stop is a no op rather than an unbalanced kernel call.
        common.start_dirty_log(&[], 0x1000).unwrap();
        assert!(!mock.dma_logging_started());
        common.stop_dirty_log().unwrap();
    }

    #[test]
    fn start_stop_dirty_log_drives_mock_when_migration_enabled() {
        let mock = MockVfio::for_dma_logging(Vec::new());
        let mut common = test_vfio_common(Arc::clone(&mock), Some(1));
        let ranges = [DmaLoggingRange {
            iova: 0x4000,
            length: 0x2000,
        }];

        common.start_dirty_log(&ranges, 0x1000).unwrap();
        assert!(mock.dma_logging_started());
        let (page_size, recorded) = mock.dma_logging_recorded();
        assert_eq!(page_size, 0x1000);
        assert_eq!(recorded, ranges);

        common.stop_dirty_log().unwrap();
        assert!(!mock.dma_logging_started());
    }

    #[test]
    fn dirty_log_uses_negotiated_page_size() {
        // The mock negotiates 0x2000 against a requested 0x1000. With bit 0
        // set the reported range length equals the negotiated granularity.
        let mock = MockVfio::with_state(MockVfioState {
            dma_logging_bitmap: vec![0b1_u64],
            dma_logging_negotiated: Some(0x2000),
            ..Default::default()
        });
        let mut common = test_vfio_common(mock, Some(1));
        let ranges = [DmaLoggingRange {
            iova: 0x1_0000_0000,
            length: 0x2000 * 64,
        }];

        common.start_dirty_log(&ranges, 0x1000).unwrap();
        let table = common.dirty_log(&ranges).unwrap();
        let merged = table.ranges();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].gpa, 0x1_0000_0000);
        assert_eq!(merged[0].length, 0x2000);
    }

    #[test]
    fn dirty_log_merges_ranges_into_one_table() {
        // Mock returns the same canned bitmap for each report call. With
        // bit 0 set, each range contributes its first page.
        let bitmap = vec![0b1_u64];
        let mock = MockVfio::for_dma_logging(bitmap);
        let mut common = test_vfio_common(mock, Some(1));
        let page_size: u64 = 0x1000;
        let ranges = [
            DmaLoggingRange {
                iova: 0x1_0000_0000,
                length: page_size * 64,
            },
            DmaLoggingRange {
                iova: 0x2_0000_0000,
                length: page_size * 64,
            },
        ];

        common.start_dirty_log(&ranges, page_size).unwrap();
        let table = common.dirty_log(&ranges).unwrap();
        let merged = table.ranges();
        assert_eq!(merged.len(), 2);
        let mut starts: Vec<u64> = merged.iter().map(|r| r.gpa).collect();
        starts.sort();
        assert_eq!(starts, vec![0x1_0000_0000, 0x2_0000_0000]);
        assert!(merged.iter().all(|r| r.length == page_size));
    }

    // pause() and resume() name no recovery state, so a failed transition
    // resets the device directly.
    #[test]
    fn transition_without_recovery_state_resets() {
        let mock = MockVfio::failing_at(&[VfioMigrationState::Running]);
        let common = test_vfio_common(Arc::clone(&mock), Some(1));
        let err = common
            .transition_migration_state_with_recovery(VfioMigrationState::Running, None)
            .unwrap_err();
        assert!(err.to_string().contains("Running"));
        assert_eq!(mock.transitions(), vec![VfioMigrationState::Running]);
        assert_eq!(mock.resets(), 1);
    }

    // A snapshot with migration state restored onto a device without migration
    // support must fail rather than silently drop the device state.
    #[test]
    fn set_state_rejects_migration_data_without_support() {
        let mock = MockVfio::for_load();
        let mut common = test_vfio_common(mock, None);
        let state = VfioCommonState {
            intx_state: None,
            msi_state: None,
            msix_state: None,
            patches: HashMap::new(),
        };
        let mig = VfioMigrationData {
            blob: String::new(),
        };
        let err = common.set_state(&state, None, None, Some(mig)).unwrap_err();
        assert!(matches!(err, VfioPciError::RestoreMigration(_)));
    }

    // A guest write to a non BAR, non MSI config register must mirror into
    // the PciConfiguration shadow so a later snapshot() picks up the live
    // value.
    #[test]
    fn write_config_register_mirrors_non_bar_into_shadow() {
        let mock = MockVfio::for_load();
        let mut common = test_vfio_common(mock, Some(1));

        // PCI_COMMAND is reg index 1. Write the 16 bit command word only.
        let cmd: u16 = 0x0406;
        common.write_config_register(COMMAND_REG, 0, &cmd.to_le_bytes());

        let got = common.configuration.read_reg(COMMAND_REG) & 0xFFFF;
        assert_eq!(got as u16, cmd);
    }

    // Extended capability handling, driven through a mock Vfio wrapper backed
    // by an in memory PCI express config space.

    struct MockConfigSpace {
        space: Mutex<Vec<u8>>,
    }

    impl MockConfigSpace {
        fn new(caps: &[(u32, PciExpressCapabilityId, u32)]) -> Arc<Self> {
            let mut space = vec![0u8; PCIE_CONFIG_SPACE_SIZE as usize];

            for (offset, id, next) in caps {
                let header = (*id as u32) | (next << PCI_EXT_CAP_NEXT_SHIFT);
                let start = *offset as usize;
                space[start..start + 4].copy_from_slice(&header.to_le_bytes());
            }

            Arc::new(MockConfigSpace {
                space: Mutex::new(space),
            })
        }

        fn dword(&self, offset: u32) -> u32 {
            let space = self.space.lock().unwrap();
            let start = offset as usize;
            u32::from_le_bytes(space[start..start + 4].try_into().unwrap())
        }
    }

    impl Vfio for MockConfigSpace {
        fn region_read(&self, _index: u32, offset: u64, data: &mut [u8]) {
            let space = self.space.lock().unwrap();
            let start = offset as usize;
            data.copy_from_slice(&space[start..start + data.len()]);
        }

        fn region_write(&self, _index: u32, offset: u64, data: &[u8]) {
            let mut space = self.space.lock().unwrap();
            let start = offset as usize;
            space[start..start + data.len()].copy_from_slice(data);
        }
    }

    fn test_vfio_common_with_pasid(vfio_wrapper: Arc<MockConfigSpace>) -> VfioCommon {
        let mut common = test_vfio_common(vfio_wrapper, None);
        common.extended_caps = vec![Arc::new(PasidCap::new(16, false, false))];
        common
    }

    #[test]
    fn extended_capability_is_appended_to_the_chain() {
        let mock = MockConfigSpace::new(&[
            (
                0x100,
                PciExpressCapabilityId::SingleRootIoVirtualization,
                0x140,
            ),
            (0x140, PciExpressCapabilityId::AdvancedErrorReporting, 0x180),
            (0x180, PciExpressCapabilityId::ResizeableBar, 0),
        ]);
        let mut common = test_vfio_common_with_pasid(mock);

        common.parse_extended_capabilities().unwrap();

        // The chain must start at 0x100, so the hidden head becomes a null
        // capability pointing at the first capability left in the chain.
        let head = common.read_config_register(0x100 / 4);
        assert_eq!(head & 0xffff, PciExpressCapabilityId::NullCapability as u32);
        assert_eq!(head >> PCI_EXT_CAP_NEXT_SHIFT, 0x140);

        // The last capability kept from the device now points at the appended
        // one, which terminates the chain.
        let pasid = PasidCap::new(16, false, false);
        let offset = PCIE_CONFIG_SPACE_SIZE - pasid.size();

        let kept = common.read_config_register(0x140 / 4);
        assert_eq!(
            kept & 0xffff,
            PciExpressCapabilityId::AdvancedErrorReporting as u32
        );
        assert_eq!(kept >> PCI_EXT_CAP_NEXT_SHIFT, offset);

        let appended = common.read_config_register((offset / 4) as usize);
        assert_eq!(
            appended & 0xffff,
            PciExpressCapabilityId::ProcessAddressSpaceId as u32
        );
        assert_eq!(appended >> PCI_EXT_CAP_NEXT_SHIFT, 0);
        assert_eq!(
            common.read_config_register((offset / 4) as usize + 1),
            pasid.dwords()[0]
        );

        // With every capability of the device hidden, the null head is the one
        // pointing at the appended capability.
        let mock = MockConfigSpace::new(&[
            (
                0x100,
                PciExpressCapabilityId::SingleRootIoVirtualization,
                0x140,
            ),
            (0x140, PciExpressCapabilityId::ResizeableBar, 0),
        ]);
        let mut common = test_vfio_common_with_pasid(mock);

        common.parse_extended_capabilities().unwrap();

        let head = common.read_config_register(0x100 / 4);
        assert_eq!(head & 0xffff, PciExpressCapabilityId::NullCapability as u32);
        assert_eq!(head >> PCI_EXT_CAP_NEXT_SHIFT, offset);
    }

    #[test]
    fn extended_capability_without_free_space_is_rejected() {
        // The appended capability would overlap a capability provided by the
        // device, even though that one is hidden from the guest.
        let mock = MockConfigSpace::new(&[
            (0x100, PciExpressCapabilityId::AdvancedErrorReporting, 0xff8),
            (0xff8, PciExpressCapabilityId::SingleRootIoVirtualization, 0),
        ]);
        let mut common = test_vfio_common_with_pasid(mock);
        assert!(matches!(
            common.parse_extended_capabilities(),
            Err(VfioPciError::ExtendedCapNoSpace(_))
        ));

        // No space is left behind the last capability of the device's chain.
        let mock = MockConfigSpace::new(&[
            (0x100, PciExpressCapabilityId::AdvancedErrorReporting, 0xffc),
            (0xffc, PciExpressCapabilityId::AdvancedErrorReporting, 0),
        ]);
        let mut common = test_vfio_common_with_pasid(mock);
        assert!(matches!(
            common.parse_extended_capabilities(),
            Err(VfioPciError::ExtendedCapNoSpace(_))
        ));
    }

    #[test]
    fn only_fully_patched_registers_are_handled_locally() {
        let mock = MockConfigSpace::new(&[]);
        let mut common = test_vfio_common(Arc::clone(&mock), None);

        // A patch covering a few fields leaves the rest of the register to the
        // device, so the write must still reach it.
        common.patch_reg(0x40 / 4, 0x0000_ffff, 0x1234, 0);
        common.write_config_register(0x40 / 4, 0, &0xabcd_5678u32.to_le_bytes());
        assert_eq!(mock.dword(0x40), 0xabcd_5678);
        assert_eq!(common.read_config_register(0x40 / 4), 0xabcd_1234);

        // A patch covering the whole register describes a register the device
        // doesn't have, so the write only updates the writable fields.
        common.patch_reg(0x50 / 4, 0xffff_ffff, 0x0000_0001, 0x0000_00ff);
        common.write_config_register(0x50 / 4, 0, &0x1111_2222u32.to_le_bytes());
        assert_eq!(mock.dword(0x50), 0);
        assert_eq!(common.read_config_register(0x50 / 4), 0x0000_0022);
    }

    #[test]
    fn patches_are_restored_from_a_snapshot() {
        let mock = MockConfigSpace::new(&[]);
        let mut common = test_vfio_common(Arc::clone(&mock), None);
        common.patch_reg(0x40 / 4, 0x0000_ffff, 0x1234, 0);

        let snapshot = Snapshot::new_from_state(&common.state()).unwrap();
        let state: VfioCommonState = snapshot.to_state().unwrap();

        let mut restored = test_vfio_common(mock, None);
        restored.set_state(&state, None, None, None).unwrap();

        assert_eq!(restored.read_config_register(0x40 / 4), 0x0000_1234);
    }

    // clique_skip_reason(): the pure fail-closed decision function
    // map_mmio_regions() delegates to under x_nv_gpudirect_clique. Each
    // row below is named after the (region index, MMAP flag, caps,
    // clique on/off) shape it exercises; no VfioDevice/fd/VM mock is
    // needed since the function takes only plain data.

    fn sparse_cap(size: u64) -> VfioRegionInfoCap {
        VfioRegionInfoCap::SparseMmap(VfioRegionInfoCapSparseMmap {
            areas: vec![VfioRegionSparseMmapArea { offset: 0, size }],
        })
    }

    #[test]
    fn test_clique_skip_reason_rom_region_is_exempt() {
        // The bug row: the expansion ROM never gets FLAG_MMAP (the
        // kernel only ever grants it FLAG_READ), yet every documented
        // clique GPU carries a VBIOS ROM - this must stay silent.
        assert!(
            VfioPciDevice::clique_skip_reason(
                true,
                VFIO_PCI_ROM_REGION_INDEX,
                PciBarRegionType::Memory32BitRegion,
                0,
                &[],
                false,
                false,
                (0, 0),
            )
            .is_none()
        );
    }

    #[test]
    fn test_clique_skip_reason_io_bar_is_exempt() {
        // I/O port BARs are never memory-mapped at all.
        assert!(
            VfioPciDevice::clique_skip_reason(
                true,
                2,
                PciBarRegionType::IoRegion,
                0,
                &[],
                false,
                false,
                (0, 0),
            )
            .is_none()
        );
    }

    #[test]
    fn test_clique_skip_reason_msix_carve_is_not_a_hole() {
        // MsixMappable is the FIRST matching cap: generate_sparse_areas()
        // takes the deliberate MSI-X table/PBA carve-out path, so the
        // "gap" implied by (mapped, size) here is not a stage-2
        // mapping defect and must not be flagged.
        let caps = [VfioRegionInfoCap::MsixMappable, sparse_cap(0x1000)];
        assert!(
            VfioPciDevice::clique_skip_reason(
                true,
                2,
                PciBarRegionType::Memory64BitRegion,
                VFIO_REGION_INFO_FLAG_MMAP,
                &caps,
                false,
                true,
                (0x1000, 0x2000),
            )
            .is_none()
        );
    }

    #[test]
    fn test_clique_skip_reason_kernel_sparse_hole_is_fatal_when_clique_on() {
        let caps = [sparse_cap(0x1000)];
        let reason = VfioPciDevice::clique_skip_reason(
            true,
            2,
            PciBarRegionType::Memory64BitRegion,
            VFIO_REGION_INFO_FLAG_MMAP,
            &caps,
            false,
            false,
            (0x1000, 0x2000),
        );
        assert!(matches!(
            reason,
            Some(CliqueSkipReason::KernelSparseHole {
                mapped: 0x1000,
                size: 0x2000,
            })
        ));
    }

    #[test]
    fn test_clique_skip_reason_kernel_sparse_hole_is_silent_when_clique_off() {
        let caps = [sparse_cap(0x1000)];
        assert!(
            VfioPciDevice::clique_skip_reason(
                false,
                2,
                PciBarRegionType::Memory64BitRegion,
                VFIO_REGION_INFO_FLAG_MMAP,
                &caps,
                false,
                false,
                (0x1000, 0x2000),
            )
            .is_none()
        );
    }

    #[test]
    fn test_clique_skip_reason_user_excluded_bar_is_fatal_when_clique_on() {
        let reason = VfioPciDevice::clique_skip_reason(
            true,
            2,
            PciBarRegionType::Memory64BitRegion,
            0,
            &[],
            true,
            false,
            (0, 0),
        );
        assert!(matches!(reason, Some(CliqueSkipReason::UserExcludedBar)));
    }

    #[test]
    fn test_clique_skip_reason_not_mmap_capable_is_fatal_when_clique_on() {
        let reason = VfioPciDevice::clique_skip_reason(
            true,
            2,
            PciBarRegionType::Memory64BitRegion,
            0,
            &[],
            false,
            false,
            (0, 0),
        );
        assert!(matches!(reason, Some(CliqueSkipReason::NotMmapCapable)));
    }

    #[test]
    fn test_clique_skip_reason_msix_not_mappable_is_fatal_when_clique_on() {
        let reason = VfioPciDevice::clique_skip_reason(
            true,
            0,
            PciBarRegionType::Memory32BitRegion,
            VFIO_REGION_INFO_FLAG_MMAP,
            &[],
            false,
            true,
            (0, 0),
        );
        assert!(matches!(reason, Some(CliqueSkipReason::MsixNotMappable)));
    }

    // Boundary rows.

    #[test]
    fn test_clique_skip_reason_no_op_when_clique_not_configured() {
        // Every other input says "fail closed"; clique_configured =
        // false must still win over all of them.
        let reason = VfioPciDevice::clique_skip_reason(
            false,
            2,
            PciBarRegionType::Memory64BitRegion,
            0,
            &[],
            true,
            true,
            (0, 0x1000),
        );
        assert!(reason.is_none());
    }

    #[test]
    fn test_clique_skip_reason_sparse_mmap_full_coverage_is_not_a_hole() {
        let caps = [sparse_cap(0x2000)];
        assert!(
            VfioPciDevice::clique_skip_reason(
                true,
                2,
                PciBarRegionType::Memory64BitRegion,
                VFIO_REGION_INFO_FLAG_MMAP,
                &caps,
                false,
                false,
                (0x2000, 0x2000),
            )
            .is_none()
        );
    }

    #[test]
    fn test_clique_skip_reason_sparse_mmap_wins_when_first() {
        // The flip side of the MSI-X-carve row: SparseMmap comes
        // FIRST this time, so generate_sparse_areas() would take the
        // genuine sparse-mmap path, and a real gap here is a genuine
        // hole regardless of MsixMappable appearing later in caps.
        let caps = [sparse_cap(0x1000), VfioRegionInfoCap::MsixMappable];
        let reason = VfioPciDevice::clique_skip_reason(
            true,
            2,
            PciBarRegionType::Memory64BitRegion,
            VFIO_REGION_INFO_FLAG_MMAP,
            &caps,
            false,
            false,
            (0x1000, 0x2000),
        );
        assert!(matches!(
            reason,
            Some(CliqueSkipReason::KernelSparseHole {
                mapped: 0x1000,
                size: 0x2000,
            })
        ));
    }

    // Pure helpers for the dma-buf-backed P2P BAR mapping cascade:
    // dma_range_for_area (the sub-page BAR skip) and
    // classify_dma_buf_errno (the fallback ladder).

    #[test]
    fn dma_range_for_area_full_bar_aligned() {
        // 16 MiB BAR at a 64 KiB-aligned iova, alignment 0x10000.
        let iova = 0x1_0000_0000u64;
        let area_size = 16 * 1024 * 1024u64;
        assert_eq!(
            dma_range_for_area(iova, area_size, 0x10000),
            Some(area_size)
        );
    }

    #[test]
    fn dma_range_for_area_sub_page_nvme_area_is_none() {
        // The measured NVMe case: a 16 KiB area cannot satisfy a 64 KiB
        // alignment contract.
        let iova = 0x1_0000_0000u64;
        let area_size = 16 * 1024u64;
        assert_eq!(dma_range_for_area(iova, area_size, 0x10000), None);
    }

    #[test]
    fn dma_range_for_area_zero_alignment_is_none() {
        // Alignment 0 means the backend offers no file-backed path at all.
        assert_eq!(dma_range_for_area(0x1_0000_0000, 16 * 1024 * 1024, 0), None);
    }

    #[test]
    fn dma_range_for_area_alignment_one_is_some() {
        // The kernel documents 1 as "any IOVA allowed".
        assert_eq!(dma_range_for_area(0x1234_5678, 0x2345, 1), Some(0x2345));
    }

    #[test]
    fn dma_range_for_area_misaligned_iova_is_none() {
        let iova = 0x1_0000_0001u64; // not a multiple of 0x10000
        let area_size = 16 * 1024 * 1024u64;
        assert_eq!(dma_range_for_area(iova, area_size, 0x10000), None);
    }

    #[test]
    fn classify_dma_buf_errno_maps_every_arm() {
        assert_eq!(
            classify_dma_buf_errno(libc::ENOTTY),
            DmaBufOutcome::Unsupported
        );
        assert_eq!(
            classify_dma_buf_errno(libc::EINVAL),
            DmaBufOutcome::Unsupported
        );
        assert_eq!(
            classify_dma_buf_errno(libc::EOPNOTSUPP),
            DmaBufOutcome::Unsupported
        );
        assert_eq!(classify_dma_buf_errno(libc::ENODEV), DmaBufOutcome::Revoked);
        assert_eq!(classify_dma_buf_errno(libc::EEXIST), DmaBufOutcome::Stale);
        assert_eq!(classify_dma_buf_errno(libc::EIO), DmaBufOutcome::Failed);
    }

    // The datum the hardware run bought: a revoked dma-buf answers ENODEV,
    // and that must be retried on the next un-revoke edge, never treated
    // as a hard failure.
    #[test]
    fn classify_dma_buf_errno_enodev_is_revoked_not_failed() {
        let outcome = classify_dma_buf_errno(libc::ENODEV);
        assert_eq!(outcome, DmaBufOutcome::Revoked);
        assert_ne!(outcome, DmaBufOutcome::Failed);
    }

    // mse_transition: the memory-space-enable edge between two COMMAND
    // register reads. Values are the exact pair the probe measured.

    #[test]
    fn mse_transition_disabled_edge() {
        assert_eq!(mse_transition(0x0103, 0x0101), Some(MseEdge::Disabled));
    }

    #[test]
    fn mse_transition_enabled_edge() {
        assert_eq!(mse_transition(0x0101, 0x0103), Some(MseEdge::Enabled));
    }

    #[test]
    fn mse_transition_unchanged_enabled_is_none() {
        assert_eq!(mse_transition(0x0103, 0x0103), None);
    }

    #[test]
    fn mse_transition_unchanged_disabled_is_none() {
        assert_eq!(mse_transition(0x0101, 0x0101), None);
    }

    #[test]
    fn mse_transition_unrelated_bit_change_is_none() {
        // Bit 2 (bus master enable) flips; MSE (bit 1) stays disabled.
        assert_eq!(mse_transition(0x0101, 0x0105), None);
    }

    // d_state_transition: the D-state edge between two PMCSR reads.

    #[test]
    fn d_state_transition_entered_low_power() {
        assert_eq!(
            d_state_transition(0x0000, 0x0003),
            Some(DStateEdge::EnteredLowPower)
        );
    }

    #[test]
    fn d_state_transition_returned_to_d0() {
        assert_eq!(
            d_state_transition(0x0003, 0x0000),
            Some(DStateEdge::ReturnedToD0)
        );
    }

    #[test]
    fn d_state_transition_unchanged_d0_is_none() {
        assert_eq!(d_state_transition(0x0000, 0x0000), None);
    }

    #[test]
    fn d_state_transition_pme_en_only_change_is_none() {
        // PME_En (bit 8) flips; the power-state field (bits 1:0) stays D0.
        assert_eq!(d_state_transition(0x0000, 0x0100), None);
    }

    // write_sets_flr: detecting a BCR_FLR-setting write from the write
    // data itself, across the byte/offset shapes a config-space write
    // can take.

    #[test]
    fn write_sets_flr_four_byte_write_with_bit_set() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        let data = 0x0000_8010u32.to_le_bytes();
        assert!(write_sets_flr(pcie_cap_offset, reg_idx, 0, &data));
    }

    #[test]
    fn write_sets_flr_two_byte_write_at_offset_zero_with_bit_set() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        let data = 0x8000u16.to_le_bytes();
        assert!(write_sets_flr(pcie_cap_offset, reg_idx, 0, &data));
    }

    #[test]
    fn write_sets_flr_one_byte_write_at_offset_one_with_bit_set() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        assert!(write_sets_flr(pcie_cap_offset, reg_idx, 1, &[0x80]));
    }

    #[test]
    fn write_sets_flr_bit_not_set_is_false() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        assert!(!write_sets_flr(pcie_cap_offset, reg_idx, 0, &[0x00, 0x00]));
    }

    #[test]
    fn write_sets_flr_wrong_register_is_false() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        assert!(!write_sets_flr(
            pcie_cap_offset,
            reg_idx + 1,
            0,
            &[0x80, 0x80]
        ));
    }

    #[test]
    fn write_sets_flr_write_does_not_reach_flr_byte_is_false() {
        let pcie_cap_offset = 0x40u8;
        let reg_idx = (pcie_cap_offset as usize + 8) / 4;
        // A 1-byte write at offset 0 touches only byte 0, never byte 1.
        assert!(!write_sets_flr(pcie_cap_offset, reg_idx, 0, &[0xff]));
        // A write starting at offset 2 doesn't reach byte 1 either.
        assert!(!write_sets_flr(pcie_cap_offset, reg_idx, 2, &[0xff, 0xff]));
    }
}
