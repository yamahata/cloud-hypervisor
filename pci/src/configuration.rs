// Copyright 2018 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::result;
use std::sync::{Arc, Mutex};

use byteorder::{ByteOrder, LittleEndian};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use vm_device::PciBarType;
use vm_migration::{MigratableError, Pausable, Snapshot, Snapshottable};

use crate::device::{BarRelocation, BarRelocationStatus, InstallParams, ReleaseParams};
use crate::{MsixConfig, PciInterruptPin};

// The number of 32bit registers in the config space, 4096 bytes.
const NUM_CONFIGURATION_REGISTERS: usize = 1024;

pub(crate) const COMMAND_REG: usize = 1;
pub(crate) const COMMAND_REG_IO_SPACE_MASK: u32 = 0x0000_0001;
pub(crate) const COMMAND_REG_MEMORY_SPACE_MASK: u32 = 0x0000_0002;
const STATUS_REG: usize = 1;
const STATUS_REG_CAPABILITIES_USED_MASK: u32 = 0x0010_0000;
const BAR0_REG: usize = 4;
const ROM_BAR_REG: usize = 12;
const ROM_BAR_IDX: usize = 6;
const BAR_IO_ADDR_MASK: u32 = 0xffff_fffc;
const BAR_MEM_ADDR_MASK: u32 = 0xffff_fff0;
const ROM_BAR_ADDR_MASK: u32 = 0xffff_f800;
const MSI_CAPABILITY_REGISTER_MASK: u32 = 0x0071_0000;
const MSIX_CAPABILITY_REGISTER_MASK: u32 = 0xc000_0000;
const NUM_BAR_REGS: usize = 6;
const CAPABILITY_LIST_HEAD_OFFSET: usize = 0x34;
const FIRST_CAPABILITY_OFFSET: usize = 0x40;
const CAPABILITY_MAX_OFFSET: usize = 192;

const INTERRUPT_LINE_PIN_REG: usize = 15;

pub const PCI_CONFIGURATION_ID: &str = "pci_configuration";

/// Represents the types of PCI headers allowed in the configuration registers.
#[derive(Copy, Clone)]
pub enum PciHeaderType {
    Device,
    Bridge,
}

/// Classes of PCI nodes.
#[derive(Copy, Clone)]
pub enum PciClassCode {
    TooOld,
    MassStorage,
    NetworkController,
    DisplayController,
    MultimediaController,
    MemoryController,
    BridgeDevice,
    SimpleCommunicationController,
    BaseSystemPeripheral,
    InputDevice,
    DockingStation,
    Processor,
    SerialBusController,
    WirelessController,
    IntelligentIoController,
    EncryptionController,
    DataAcquisitionSignalProcessing,
    Other = 0xff,
}

impl PciClassCode {
    pub fn get_register_value(self) -> u8 {
        self as u8
    }
}

/// A PCI subclass. Each class in `PciClassCode` can specify a unique set of subclasses. This trait
/// is implemented by each subclass. It allows use of a trait object to generate configurations.
pub trait PciSubclass {
    /// Convert this subclass to the value used in the PCI specification.
    fn get_register_value(&self) -> u8;
}

/// Subclasses of the MultimediaController class.
#[expect(dead_code)]
#[derive(Copy, Clone)]
pub enum PciMultimediaSubclass {
    VideoController = 0x00,
    AudioController = 0x01,
    TelephonyDevice = 0x02,
    AudioDevice = 0x03,
    Other = 0x80,
}

impl PciSubclass for PciMultimediaSubclass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

/// Subclasses of the BridgeDevice
#[expect(dead_code)]
#[derive(Copy, Clone)]
pub enum PciBridgeSubclass {
    HostBridge = 0x00,
    IsaBridge = 0x01,
    EisaBridge = 0x02,
    McaBridge = 0x03,
    PciToPciBridge = 0x04,
    PcmciaBridge = 0x05,
    NuBusBridge = 0x06,
    CardBusBridge = 0x07,
    RacEwayBridge = 0x08,
    PciToPciSemiTransparentBridge = 0x09,
    InfiniBrandToPciHostBridge = 0x0a,
    OtherBridgeDevice = 0x80,
}

impl PciSubclass for PciBridgeSubclass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

/// Subclass of the SerialBus
#[derive(Copy, Clone)]
pub enum PciSerialBusSubClass {
    Firewire = 0x00,
    Accessbus = 0x01,
    Ssa = 0x02,
    Usb = 0x03,
}

impl PciSubclass for PciSerialBusSubClass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

/// Mass Storage Sub Classes
#[derive(Copy, Clone)]
pub enum PciMassStorageSubclass {
    ScsiStorage = 0x00,
    IdeInterface = 0x01,
    FloppyController = 0x02,
    IpiController = 0x03,
    RaidController = 0x04,
    AtaController = 0x05,
    SataController = 0x06,
    SerialScsiController = 0x07,
    NvmController = 0x08,
    MassStorage = 0x80,
}

impl PciSubclass for PciMassStorageSubclass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

/// Network Controller Sub Classes
#[derive(Copy, Clone)]
pub enum PciNetworkControllerSubclass {
    EthernetController = 0x00,
    TokenRingController = 0x01,
    FddiController = 0x02,
    AtmController = 0x03,
    IsdnController = 0x04,
    WorldFipController = 0x05,
    PicmgController = 0x06,
    InfinibandController = 0x07,
    FabricController = 0x08,
    NetworkController = 0x80,
}

impl PciSubclass for PciNetworkControllerSubclass {
    fn get_register_value(&self) -> u8 {
        *self as u8
    }
}

/// Trait to define a PCI class programming interface
///
/// Each combination of `PciClassCode` and `PciSubclass` can specify a
/// set of register-level programming interfaces.
/// This trait is implemented by each programming interface.
/// It allows use of a trait object to generate configurations.
pub trait PciProgrammingInterface {
    /// Convert this programming interface to the value used in the PCI specification.
    fn get_register_value(&self) -> u8;
}

/// Types of PCI capabilities.
#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(u8)]
pub enum PciCapabilityId {
    ListId = 0,
    PowerManagement = 0x01,
    AcceleratedGraphicsPort = 0x02,
    VitalProductData = 0x03,
    SlotIdentification = 0x04,
    MessageSignalledInterrupts = 0x05,
    CompactPciHotSwap = 0x06,
    PciX = 0x07,
    HyperTransport = 0x08,
    VendorSpecific = 0x09,
    Debugport = 0x0A,
    CompactPciCentralResourceControl = 0x0B,
    PciStandardHotPlugController = 0x0C,
    BridgeSubsystemVendorDeviceId = 0x0D,
    AgpTargetPciPcibridge = 0x0E,
    SecureDevice = 0x0F,
    PciExpress = 0x10,
    MsiX = 0x11,
    SataDataIndexConf = 0x12,
    PciAdvancedFeatures = 0x13,
    PciEnhancedAllocation = 0x14,
}

impl From<u8> for PciCapabilityId {
    fn from(c: u8) -> Self {
        match c {
            0 => PciCapabilityId::ListId,
            0x01 => PciCapabilityId::PowerManagement,
            0x02 => PciCapabilityId::AcceleratedGraphicsPort,
            0x03 => PciCapabilityId::VitalProductData,
            0x04 => PciCapabilityId::SlotIdentification,
            0x05 => PciCapabilityId::MessageSignalledInterrupts,
            0x06 => PciCapabilityId::CompactPciHotSwap,
            0x07 => PciCapabilityId::PciX,
            0x08 => PciCapabilityId::HyperTransport,
            0x09 => PciCapabilityId::VendorSpecific,
            0x0A => PciCapabilityId::Debugport,
            0x0B => PciCapabilityId::CompactPciCentralResourceControl,
            0x0C => PciCapabilityId::PciStandardHotPlugController,
            0x0D => PciCapabilityId::BridgeSubsystemVendorDeviceId,
            0x0E => PciCapabilityId::AgpTargetPciPcibridge,
            0x0F => PciCapabilityId::SecureDevice,
            0x10 => PciCapabilityId::PciExpress,
            0x11 => PciCapabilityId::MsiX,
            0x12 => PciCapabilityId::SataDataIndexConf,
            0x13 => PciCapabilityId::PciAdvancedFeatures,
            0x14 => PciCapabilityId::PciEnhancedAllocation,
            _ => PciCapabilityId::ListId,
        }
    }
}

/// Types of PCI Express capabilities.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[repr(u16)]
pub enum PciExpressCapabilityId {
    NullCapability = 0x0000,
    AdvancedErrorReporting = 0x0001,
    VirtualChannelMultiFunctionVirtualChannelNotPresent = 0x0002,
    DeviceSerialNumber = 0x0003,
    PowerBudgeting = 0x0004,
    RootComplexLinkDeclaration = 0x0005,
    RootComplexInternalLinkControl = 0x0006,
    RootComplexEventCollectorEndpointAssociation = 0x0007,
    MultiFunctionVirtualChannel = 0x0008,
    VirtualChannelMultiFunctionVirtualChannelPresent = 0x0009,
    RootComplexRegisterBlock = 0x000a,
    VendorSpecificExtendedCapability = 0x000b,
    ConfigurationAccessCorrelation = 0x000c,
    AccessControlServices = 0x000d,
    AlternativeRoutingIdentificationInterpretation = 0x000e,
    AddressTranslationServices = 0x000f,
    SingleRootIoVirtualization = 0x0010,
    DeprecatedMultiRootIoVirtualization = 0x0011,
    Multicast = 0x0012,
    PageRequestInterface = 0x0013,
    ReservedForAmd = 0x0014,
    ResizeableBar = 0x0015,
    DynamicPowerAllocation = 0x0016,
    ThpRequester = 0x0017,
    LatencyToleranceReporting = 0x0018,
    SecondaryPciExpress = 0x0019,
    ProtocolMultiplexing = 0x001a,
    ProcessAddressSpaceId = 0x001b,
    LnRequester = 0x001c,
    DownstreamPortContainment = 0x001d,
    L1PmSubstates = 0x001e,
    PrecisionTimeMeasurement = 0x001f,
    PciExpressOverMphy = 0x0020,
    FRSQueueing = 0x0021,
    ReadinessTimeReporting = 0x0022,
    DesignatedVendorSpecificExtendedCapability = 0x0023,
    VfResizeableBar = 0x0024,
    DataLinkFeature = 0x0025,
    PhysicalLayerSixteenGts = 0x0026,
    LaneMarginingAtTheReceiver = 0x0027,
    HierarchyId = 0x0028,
    NativePcieEnclosureManagement = 0x0029,
    PhysicalLayerThirtyTwoGts = 0x002a,
    AlternateProtocol = 0x002b,
    SystemFirmwareIntermediary = 0x002c,
    ShadowFunctions = 0x002d,
    DataObjectExchange = 0x002e,
    Reserved = 0x002f,
    ExtendedCapabilitiesAbsence = 0xffff,
}

impl From<u16> for PciExpressCapabilityId {
    fn from(c: u16) -> Self {
        match c {
            0x0000 => PciExpressCapabilityId::NullCapability,
            0x0001 => PciExpressCapabilityId::AdvancedErrorReporting,
            0x0002 => PciExpressCapabilityId::VirtualChannelMultiFunctionVirtualChannelNotPresent,
            0x0003 => PciExpressCapabilityId::DeviceSerialNumber,
            0x0004 => PciExpressCapabilityId::PowerBudgeting,
            0x0005 => PciExpressCapabilityId::RootComplexLinkDeclaration,
            0x0006 => PciExpressCapabilityId::RootComplexInternalLinkControl,
            0x0007 => PciExpressCapabilityId::RootComplexEventCollectorEndpointAssociation,
            0x0008 => PciExpressCapabilityId::MultiFunctionVirtualChannel,
            0x0009 => PciExpressCapabilityId::VirtualChannelMultiFunctionVirtualChannelPresent,
            0x000a => PciExpressCapabilityId::RootComplexRegisterBlock,
            0x000b => PciExpressCapabilityId::VendorSpecificExtendedCapability,
            0x000c => PciExpressCapabilityId::ConfigurationAccessCorrelation,
            0x000d => PciExpressCapabilityId::AccessControlServices,
            0x000e => PciExpressCapabilityId::AlternativeRoutingIdentificationInterpretation,
            0x000f => PciExpressCapabilityId::AddressTranslationServices,
            0x0010 => PciExpressCapabilityId::SingleRootIoVirtualization,
            0x0011 => PciExpressCapabilityId::DeprecatedMultiRootIoVirtualization,
            0x0012 => PciExpressCapabilityId::Multicast,
            0x0013 => PciExpressCapabilityId::PageRequestInterface,
            0x0014 => PciExpressCapabilityId::ReservedForAmd,
            0x0015 => PciExpressCapabilityId::ResizeableBar,
            0x0016 => PciExpressCapabilityId::DynamicPowerAllocation,
            0x0017 => PciExpressCapabilityId::ThpRequester,
            0x0018 => PciExpressCapabilityId::LatencyToleranceReporting,
            0x0019 => PciExpressCapabilityId::SecondaryPciExpress,
            0x001a => PciExpressCapabilityId::ProtocolMultiplexing,
            0x001b => PciExpressCapabilityId::ProcessAddressSpaceId,
            0x001c => PciExpressCapabilityId::LnRequester,
            0x001d => PciExpressCapabilityId::DownstreamPortContainment,
            0x001e => PciExpressCapabilityId::L1PmSubstates,
            0x001f => PciExpressCapabilityId::PrecisionTimeMeasurement,
            0x0020 => PciExpressCapabilityId::PciExpressOverMphy,
            0x0021 => PciExpressCapabilityId::FRSQueueing,
            0x0022 => PciExpressCapabilityId::ReadinessTimeReporting,
            0x0023 => PciExpressCapabilityId::DesignatedVendorSpecificExtendedCapability,
            0x0024 => PciExpressCapabilityId::VfResizeableBar,
            0x0025 => PciExpressCapabilityId::DataLinkFeature,
            0x0026 => PciExpressCapabilityId::PhysicalLayerSixteenGts,
            0x0027 => PciExpressCapabilityId::LaneMarginingAtTheReceiver,
            0x0028 => PciExpressCapabilityId::HierarchyId,
            0x0029 => PciExpressCapabilityId::NativePcieEnclosureManagement,
            0x002a => PciExpressCapabilityId::PhysicalLayerThirtyTwoGts,
            0x002b => PciExpressCapabilityId::AlternateProtocol,
            0x002c => PciExpressCapabilityId::SystemFirmwareIntermediary,
            0x002d => PciExpressCapabilityId::ShadowFunctions,
            0x002e => PciExpressCapabilityId::DataObjectExchange,
            0xffff => PciExpressCapabilityId::ExtendedCapabilitiesAbsence,
            _ => PciExpressCapabilityId::Reserved,
        }
    }
}

/// A PCI capability list. Devices can optionally specify capabilities in their configuration space.
pub trait PciCapability {
    fn bytes(&self) -> &[u8];
    fn id(&self) -> PciCapabilityId;
}

fn encode_32_bits_bar_size(bar_size: u32) -> Option<u32> {
    if bar_size > 0 {
        return Some(!(bar_size - 1));
    }
    None
}

fn decode_32_bits_bar_size(bar_size: u32) -> Option<u32> {
    if bar_size > 0 {
        return Some(!bar_size + 1);
    }
    None
}

fn encode_64_bits_bar_size(bar_size: u64) -> Option<(u32, u32)> {
    if bar_size > 0 {
        let result = !(bar_size - 1);
        let result_hi = (result >> 32) as u32;
        let result_lo = (result & 0xffff_ffff) as u32;
        return Some((result_hi, result_lo));
    }
    None
}

fn decode_64_bits_bar_size(bar_size_hi: u32, bar_size_lo: u32) -> Option<u64> {
    let bar_size: u64 = ((bar_size_hi as u64) << 32) | (bar_size_lo as u64);
    if bar_size > 0 {
        return Some(!bar_size + 1);
    }
    None
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct PciBar {
    addr: u32,
    size: u32,
    used: bool,
    r#type: Option<PciBarRegionType>,
}

#[derive(Serialize, Deserialize)]
pub struct PciConfigurationState {
    registers: Vec<u32>,
    writable_bits: Vec<u32>,
    bars: Vec<PciBar>,
    rom_bar_addr: u32,
    rom_bar_size: u32,
    rom_bar_used: bool,
    last_capability: Option<(usize, usize)>,
    msix_cap_reg_idx: Option<usize>,
    // Preserve deferred BAR moves across snapshot and restore. Each entry
    // records a BAR that was released (old_base) and not yet re-installed at
    // its config-space target (new_base). The field predates this fix
    // (db93c6fdc), so snapshots cross both directions: old snapshots restore
    // on this code, and snapshots taken here replay on old binaries.
    #[serde(default)]
    pending_bar_reprogram: Vec<BarReprogrammingParams>,
}

/// Legacy wire shape of an in-flight BAR move in
/// [`PciConfigurationState::pending_bar_reprogram`], kept for cross-version
/// snapshots (an old binary replays it as `old_base` -> `new_base`), and
/// confined here so nothing outside the snapshot path depends on it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct BarReprogrammingParams {
    /// The BAR slot being reprogrammed (expansion ROM = ROM_BAR_IDX; the
    /// low/primary slot for a 64-bit BAR). Identifies the BAR across the
    /// whole relocation, unlike the addresses, which a later config write
    /// can change again. Old snapshots (pre-index-keying) don't carry
    /// this field; it only matters for in-flight moves.
    #[serde(default)]
    bar_idx: usize,
    old_base: u64,
    new_base: u64,
    len: u64,
    region_type: PciBarRegionType,
}

/// Contains the configuration space of a PCI node.
///
/// See the [specification](https://en.wikipedia.org/wiki/PCI_configuration_space).
/// The configuration space is accessed with DWORD reads and writes from the guest.
pub struct PciConfiguration {
    registers: [u32; NUM_CONFIGURATION_REGISTERS],
    writable_bits: [u32; NUM_CONFIGURATION_REGISTERS], // writable bits for each register.
    bars: [PciBar; NUM_BAR_REGS],
    rom_bar_addr: u32,
    rom_bar_size: u32,
    rom_bar_used: bool,
    // Contains the byte offset and size of the last capability.
    last_capability: Option<(usize, usize)>,
    msix_cap_reg_idx: Option<usize>,
    msix_config: Option<Arc<Mutex<MsixConfig>>>,
    // BAR relocation state machine.
    //
    // A guest relocates a BAR by rewriting its address register(s) and then
    // toggling the COMMAND decode bit. To keep a multi-BAR swap from hitting a
    // free-then-allocate overlap, the old location is released eagerly on the
    // address write and the new one installed later, once decode permits, so
    // every release precedes every install. Each BAR slot (ROM at ROM_BAR_IDX)
    // is then in one of four states:
    //
    //   BOOT:        mapped_addr = None, pending_relocation = None
    //                -- the slot at construction, before add_pci_bar /
    //                   add_pci_rom_bar seeds it (also an unused BAR slot).
    //   MAPPED(A):   mapped_addr = Some(A), pending_relocation = None
    //                -- live at A (allocator + bus + guest-physical mapping).
    //   RELEASED(r): mapped_addr = None,    pending_relocation = Some(r)
    //                -- old range at r torn down, nothing mapped, awaiting
    //                   install at the address the guest last wrote.
    //   RESTORED_INFLIGHT(r): mapped_addr = Some(r), pending_relocation = Some(r)
    //                -- a snapshot restored mid-move: materialized at the old
    //                   base r (ranges live) with the move still pending; the
    //                   next decode edge replays it (release r, then install the
    //                   target). Still mapped, so NOT "released".
    //
    // A BAR's address is recorded in four places that DIVERGE while a move is in
    // flight; `is_bar_released` reports the released state to teardown paths
    // (t = the new target the guest just wrote):
    //
    //   structure               BOOT   MAPPED(A)   RELEASED(r)  RESTORED_INFLIGHT(r)
    //   ----------------------  -----  ----------  -----------  --------------------
    //   bars[slot].addr         0      A           t            t
    //   bar_regions (device)    -      A           r            r
    //   mapped_addr[slot]       None   Some(A)     None         Some(r)
    //   pending_relocation      None   None        Some(r)      Some(r)
    //   is_bar_released()       false  false       true         false
    //
    //   * bars[slot].addr    -- config-space register shadow; follows the
    //     guest's request, so it jumps to the new target t the instant the
    //     guest writes, ahead of the actual mapping.
    //   * bar_regions        -- the device's own record of where the BAR is
    //     installed (per-device `bar_regions`, VFIO `mmio_regions[].start`, the
    //     device_tree PciBar resource base); kept at the old base r, its ranges
    //     torn down, until the install commits.
    //   * mapped_addr[slot]  -- this layer's copy of the installed address,
    //     advanced ONLY on a successful install, so it never names an address
    //     the BAR is not actually mapped at.
    //   * pending_relocation -- the released-from base r while a move is
    //     pending; it also pins the PCI segment the install must stay within.
    //   * is_bar_released()  -- `pending.is_some() && mapped.is_none()`: true
    //     only in RELEASED, so teardown (`free_bars`) skips a torn-down BAR.
    //     RESTORED_INFLIGHT is pending yet still mapped, so it reports false and
    //     is NOT skipped.
    //
    // `add_pci_bar` / `add_pci_rom_bar` seed BOOT -> MAPPED at fresh boot;
    // `PciConfiguration::new` reseeds a restored snapshot (a mid-move BAR to
    // RESTORED_INFLIGHT). Runtime transitions live in `write_config_register`
    // (release, plus the immediate move when decode is already on), the
    // decode-edge drain (emits the deferred install, replaying a
    // RESTORED_INFLIGHT release first) and `on_bar_relocation_status` (Applied
    // advances `mapped_addr`; Pending leaves the slot RELEASED to retry). A
    // 64-bit BAR is tracked only on its low/primary slot.
    //
    // `mapped_addr[slot]` is the guest-physical address the BAR is currently
    // mapped at (allocator + bus + guest-physical mapping), or `None` while
    // the BAR is released awaiting install. For a 64-bit BAR the bookkeeping
    // lives only on the LOW/primary slot (the one carrying `r#type`), exactly
    // as seeded by `add_pci_bar`; the high slot stays `None`. `mapped_addr`
    // follows relocation *outcomes*: it is updated to a new address only when
    // an install actually succeeded (see `on_bar_relocation_status`).
    mapped_addr: [Option<u64>; NUM_BAR_REGS],
    // Same as `mapped_addr` but for the expansion ROM BAR.
    rom_mapped_addr: Option<u64>,
    // BAR slots that were released and not yet re-installed, mapped to the
    // address they were released from. Installed on the 0->1 transition of
    // the enable bit for each BAR's space (IOSE for IO
    // BARs, MSE for memory BARs and the ROM). Indexed by slot (ROM at
    // ROM_BAR_IDX), `None` where no move is pending.
    pending_relocation: [Option<u64>; NUM_BAR_REGS + 1],
}

/// See pci_regs.h in kernel
#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum PciBarRegionType {
    Memory32BitRegion = 0,
    IoRegion = 0x01,
    Memory64BitRegion = 0x04,
}

impl From<PciBarType> for PciBarRegionType {
    fn from(type_: PciBarType) -> Self {
        match type_ {
            PciBarType::Io => PciBarRegionType::IoRegion,
            PciBarType::Mmio32 => PciBarRegionType::Memory32BitRegion,
            PciBarType::Mmio64 => PciBarRegionType::Memory64BitRegion,
        }
    }
}

impl From<PciBarRegionType> for PciBarType {
    fn from(val: PciBarRegionType) -> Self {
        match val {
            PciBarRegionType::IoRegion => PciBarType::Io,
            PciBarRegionType::Memory32BitRegion => PciBarType::Mmio32,
            PciBarRegionType::Memory64BitRegion => PciBarType::Mmio64,
        }
    }
}

#[derive(Copy, Clone)]
pub enum PciBarPrefetchable {
    NotPrefetchable = 0,
    Prefetchable = 0x08,
}

impl From<PciBarPrefetchable> for bool {
    fn from(val: PciBarPrefetchable) -> Self {
        match val {
            PciBarPrefetchable::NotPrefetchable => false,
            PciBarPrefetchable::Prefetchable => true,
        }
    }
}

#[derive(Copy, Clone)]
pub struct PciBarConfiguration {
    addr: u64,
    size: u64,
    idx: usize,
    region_type: PciBarRegionType,
    prefetchable: PciBarPrefetchable,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("address {0} size {1} too big")]
    BarAddressInvalid(u64, u64),
    #[error("bar {0} already used")]
    BarInUse(usize),
    #[error("64bit bar {0} already used (requires two regs)")]
    BarInUse64(usize),
    #[error("bar {0} invalid, max {max}", max = NUM_BAR_REGS - 1)]
    BarInvalid(usize),
    #[error("64bitbar {0} invalid, requires two regs, max {max}", max = NUM_BAR_REGS - 1)]
    BarInvalid64(usize),
    #[error("bar address {0} not a power of two")]
    BarSizeInvalid(u64),
    #[error("empty capabilities are invalid")]
    CapabilityEmpty,
    #[error("Invalid capability length {0}")]
    CapabilityLengthInvalid(usize),
    #[error("capability of size {0} doesn't fit")]
    CapabilitySpaceFull(usize),
    #[error("failed to decode 32 bits BAR size")]
    Decode32BarSize,
    #[error("failed to decode 64 bits BAR size")]
    Decode64BarSize,
    #[error("failed to encode 32 bits BAR size")]
    Encode32BarSize,
    #[error("failed to encode 64 bits BAR size")]
    Encode64BarSize,
    #[error("address {0} size {1} too big")]
    RomBarAddressInvalid(u64, u64),
    #[error("rom bar {0} already used")]
    RomBarInUse(usize),
    #[error("rom bar {0} invalid, max {max}", max = NUM_BAR_REGS - 1)]
    RomBarInvalid(usize),
    #[error("rom bar address {0} not a power of two")]
    RomBarSizeInvalid(u64),
}
pub type Result<T> = result::Result<T, Error>;

/// A BAR address change detected on a config-register write: the slot, the
/// new target read from the just-written registers, and the BAR's size and
/// type. Purely an internal detection result -- the release side takes the
/// released-from address from `mapped_addr`, so no old_base travels here.
struct BarReprogramming {
    bar_idx: usize,
    new_base: u64,
    len: u64,
    region_type: PciBarRegionType,
}

impl PciConfiguration {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        vendor_id: u16,
        device_id: u16,
        revision_id: u8,
        class_code: PciClassCode,
        subclass: &dyn PciSubclass,
        programming_interface: Option<&dyn PciProgrammingInterface>,
        header_type: PciHeaderType,
        subsystem_vendor_id: u16,
        subsystem_id: u16,
        msix_config: Option<Arc<Mutex<MsixConfig>>>,
        state: Option<PciConfigurationState>,
    ) -> Self {
        let (
            registers,
            writable_bits,
            bars,
            rom_bar_addr,
            rom_bar_size,
            rom_bar_used,
            last_capability,
            msix_cap_reg_idx,
            pending_bar_reprogram,
        ) = if let Some(state) = state {
            (
                state.registers.try_into().unwrap(),
                state.writable_bits.try_into().unwrap(),
                state.bars.try_into().unwrap(),
                state.rom_bar_addr,
                state.rom_bar_size,
                state.rom_bar_used,
                state.last_capability,
                state.msix_cap_reg_idx,
                Some(state.pending_bar_reprogram),
            )
        } else {
            let mut registers = [0u32; NUM_CONFIGURATION_REGISTERS];
            let mut writable_bits = [0u32; NUM_CONFIGURATION_REGISTERS];
            registers[0] = (u32::from(device_id) << 16) | u32::from(vendor_id);
            // TODO(dverkamp): Status should be write-1-to-clear
            writable_bits[1] = 0x0000_ffff; // Status (r/o), command (r/w)
            let pi = if let Some(pi) = programming_interface {
                pi.get_register_value()
            } else {
                0
            };
            registers[2] = (u32::from(class_code.get_register_value()) << 24)
                | (u32::from(subclass.get_register_value()) << 16)
                | (u32::from(pi) << 8)
                | u32::from(revision_id);
            writable_bits[3] = 0x0000_00ff; // Cacheline size (r/w)
            match header_type {
                PciHeaderType::Device => {
                    registers[3] = 0x0000_0000; // Header type 0 (device)
                    writable_bits[15] = 0x0000_00ff; // Interrupt line (r/w)
                }
                PciHeaderType::Bridge => {
                    registers[3] = 0x0001_0000; // Header type 1 (bridge)
                    writable_bits[9] = 0xfff0_fff0; // Memory base and limit
                    writable_bits[15] = 0xffff_00ff; // Bridge control (r/w), interrupt line (r/w)
                }
            }
            registers[11] = (u32::from(subsystem_id) << 16) | u32::from(subsystem_vendor_id);

            (
                registers,
                writable_bits,
                [PciBar::default(); NUM_BAR_REGS],
                0,
                0,
                false,
                None,
                None,
                None,
            )
        };

        let mut config = PciConfiguration {
            registers,
            writable_bits,
            bars,
            rom_bar_addr,
            rom_bar_size,
            rom_bar_used,
            last_capability,
            msix_cap_reg_idx,
            msix_config,
            // Seeded below when restoring; empty at fresh boot (the
            // `add_pci_bar`/`add_pci_rom_bar` calls that follow seed these).
            mapped_addr: [None; NUM_BAR_REGS],
            rom_mapped_addr: None,
            pending_relocation: [None; NUM_BAR_REGS + 1],
        };

        // Restoring from a snapshot: reconstruct the mapping bookkeeping.
        if let Some(pending_bar_reprogram) = pending_bar_reprogram {
            // Restore maps every BAR at its device-tree resource address,
            // which equals the config-space address except for BARs with an
            // in-flight move (the resource is only updated when a move
            // commits). Seed "every declared BAR mapped at its config
            // address, nothing pending" first, then overlay the serialized
            // in-flight moves.
            for bar_num in 0..NUM_BAR_REGS {
                // A BAR's primary slot is the only one carrying `r#type`;
                // the high half of a 64-bit BAR is `used` but stays `None`,
                // matching `add_pci_bar`'s boot-time seeding.
                if config.bars[bar_num].r#type.is_some() {
                    config.mapped_addr[bar_num] = Some(config.get_bar_addr(bar_num));
                }
            }
            if config.rom_bar_used {
                config.rom_mapped_addr = Some(u64::from(config.rom_bar_addr & ROM_BAR_ADDR_MASK));
            }

            // A serialized in-flight move means the BAR was restored at the
            // address it was RELEASED from (`old_base`, the resource
            // address), not at the config-space target the guest wrote. Seed
            // `mapped_addr` with the released-from base and mark the slot
            // pending, so the guest's decode-enable edge replays the move
            // (release at old_base, install at the config target).
            for params in pending_bar_reprogram {
                if let Some(slot) = config.resolve_pending_slot(&params) {
                    config.set_mapped_addr(slot, Some(params.old_base));
                    config.pending_relocation[slot] = Some(params.old_base);
                } else {
                    warn!(
                        "Dropping restored in-flight BAR move {params:x?}: no declared BAR \
                         matches its config-space target"
                    );
                }
            }
        }

        config
    }

    /// Maps a restored `pending_bar_reprogram` entry to the BAR slot it
    /// describes. Trust `bar_idx` when it is consistent with the entry;
    /// legacy snapshots (pre-index-keying) default it to 0, so fall back to
    /// scanning for the slot whose config-space target matches the recorded
    /// move.
    fn resolve_pending_slot(&self, params: &BarReprogrammingParams) -> Option<usize> {
        let matches = |slot: usize| {
            self.bar_region_type(slot) == Some(params.region_type)
                && self.bar_target(slot) == params.new_base
        };

        if matches(params.bar_idx) {
            return Some(params.bar_idx);
        }

        (0..NUM_BAR_REGS).chain([ROM_BAR_IDX]).find(|&s| matches(s))
    }

    fn state(&self) -> PciConfigurationState {
        PciConfigurationState {
            registers: self.registers.to_vec(),
            writable_bits: self.writable_bits.to_vec(),
            bars: self.bars.to_vec(),
            rom_bar_addr: self.rom_bar_addr,
            rom_bar_size: self.rom_bar_size,
            rom_bar_used: self.rom_bar_used,
            last_capability: self.last_capability,
            msix_cap_reg_idx: self.msix_cap_reg_idx,
            // Serialize the in-flight moves in the legacy wire shape:
            // released-from base (old_base) to current config-space target
            // (new_base). Old binaries replay these entries as-is.
            pending_bar_reprogram: self
                .pending_relocation
                .iter()
                .enumerate()
                .filter_map(|(slot, &released_from)| {
                    released_from.map(|released_from| BarReprogrammingParams {
                        bar_idx: slot,
                        old_base: released_from,
                        new_base: self.bar_target(slot),
                        len: self.bar_len(slot).unwrap_or(0),
                        region_type: self
                            .bar_region_type(slot)
                            .unwrap_or(PciBarRegionType::Memory32BitRegion),
                    })
                })
                .collect(),
        }
    }

    /// Reads a 32bit register from `reg_idx` in the register map.
    pub fn read_reg(&self, reg_idx: usize) -> u32 {
        *(self.registers.get(reg_idx).unwrap_or(&0xffff_ffff))
    }

    /// Writes a 32bit register to `reg_idx` in the register map.
    pub fn write_reg(&mut self, reg_idx: usize, value: u32) {
        let mut mask = self.writable_bits[reg_idx];

        if (BAR0_REG..BAR0_REG + NUM_BAR_REGS).contains(&reg_idx) {
            // Handle very specific case where the BAR is being written with
            // all 1's to retrieve the BAR size during next BAR reading.
            if value == 0xffff_ffff {
                mask &= self.bars[reg_idx - 4].size;
            }
        } else if reg_idx == ROM_BAR_REG {
            // Handle very specific case where the BAR is being written with
            // all 1's on bits 31-11 to retrieve the BAR size during next BAR
            // reading.
            if value & ROM_BAR_ADDR_MASK == ROM_BAR_ADDR_MASK {
                mask &= self.rom_bar_size;
            }
        }

        if let Some(r) = self.registers.get_mut(reg_idx) {
            *r = (*r & !self.writable_bits[reg_idx]) | (value & mask);
        } else {
            warn!("bad PCI register write {reg_idx}");
        }
    }

    /// Writes a 16bit word to `offset`. `offset` must be 16bit aligned.
    pub fn write_word(&mut self, offset: usize, value: u16) {
        let shift = match offset % 4 {
            0 => 0,
            2 => 16,
            _ => {
                warn!("bad PCI config write offset {offset}");
                return;
            }
        };
        let reg_idx = offset / 4;

        if let Some(r) = self.registers.get_mut(reg_idx) {
            let writable_mask = self.writable_bits[reg_idx];
            let mask = (0xffffu32 << shift) & writable_mask;
            let shifted_value = (u32::from(value) << shift) & writable_mask;
            *r = *r & !mask | shifted_value;
        } else {
            warn!("bad PCI config write offset {offset}");
        }
    }

    /// Writes a byte to `offset`.
    pub fn write_byte(&mut self, offset: usize, value: u8) {
        self.write_byte_internal(offset, value, true);
    }

    /// Writes a byte to `offset`, optionally enforcing read-only bits.
    fn write_byte_internal(&mut self, offset: usize, value: u8, apply_writable_mask: bool) {
        let shift = (offset % 4) * 8;
        let reg_idx = offset / 4;

        if let Some(r) = self.registers.get_mut(reg_idx) {
            let writable_mask = if apply_writable_mask {
                self.writable_bits[reg_idx]
            } else {
                0xffff_ffff
            };
            let mask = (0xffu32 << shift) & writable_mask;
            let shifted_value = (u32::from(value) << shift) & writable_mask;
            *r = *r & !mask | shifted_value;
        } else {
            warn!("bad PCI config write offset {offset}");
        }
    }

    /// Adds a region specified by `config`.  Configures the specified BAR(s) to
    /// report this region and size to the guest kernel.  Enforces a few constraints
    /// (i.e, region size must be power of two, register not already used).
    pub fn add_pci_bar(&mut self, config: &PciBarConfiguration) -> Result<()> {
        let bar_idx = config.idx;
        let reg_idx = BAR0_REG + bar_idx;

        if self.bars[bar_idx].used {
            return Err(Error::BarInUse(bar_idx));
        }

        if !config.size.is_power_of_two() {
            return Err(Error::BarSizeInvalid(config.size));
        }

        if bar_idx >= NUM_BAR_REGS {
            return Err(Error::BarInvalid(bar_idx));
        }

        let end_addr = config
            .addr
            .checked_add(config.size - 1)
            .ok_or(Error::BarAddressInvalid(config.addr, config.size))?;
        match config.region_type {
            PciBarRegionType::Memory32BitRegion | PciBarRegionType::IoRegion => {
                if end_addr > u64::from(u32::MAX) {
                    return Err(Error::BarAddressInvalid(config.addr, config.size));
                }

                // Encode the BAR size as expected by the software running in
                // the guest.
                self.bars[bar_idx].size =
                    encode_32_bits_bar_size(config.size as u32).ok_or(Error::Encode32BarSize)?;
            }
            PciBarRegionType::Memory64BitRegion => {
                if bar_idx + 1 >= NUM_BAR_REGS {
                    return Err(Error::BarInvalid64(bar_idx));
                }

                if self.bars[bar_idx + 1].used {
                    return Err(Error::BarInUse64(bar_idx));
                }

                // Encode the BAR size as expected by the software running in
                // the guest.
                let (bar_size_hi, bar_size_lo) =
                    encode_64_bits_bar_size(config.size).ok_or(Error::Encode64BarSize)?;

                self.registers[reg_idx + 1] = (config.addr >> 32) as u32;
                self.writable_bits[reg_idx + 1] = 0xffff_ffff;
                self.bars[bar_idx + 1].addr = self.registers[reg_idx + 1];
                self.bars[bar_idx].size = bar_size_lo;
                self.bars[bar_idx + 1].size = bar_size_hi;
                self.bars[bar_idx + 1].used = true;
            }
        }

        let (mask, lower_bits) = match config.region_type {
            PciBarRegionType::Memory32BitRegion | PciBarRegionType::Memory64BitRegion => (
                BAR_MEM_ADDR_MASK,
                config.prefetchable as u32 | config.region_type as u32,
            ),
            PciBarRegionType::IoRegion => (BAR_IO_ADDR_MASK, config.region_type as u32),
        };

        self.registers[reg_idx] = ((config.addr as u32) & mask) | lower_bits;
        self.writable_bits[reg_idx] = mask;
        self.bars[bar_idx].addr = self.registers[reg_idx];
        self.bars[bar_idx].used = true;
        self.bars[bar_idx].r#type = Some(config.region_type);

        // The BAR is mapped (allocator/bus/guest-physical mapping) at its
        // initial address.
        self.mapped_addr[bar_idx] = Some(config.addr);

        Ok(())
    }

    /// Adds rom expansion BAR.
    pub fn add_pci_rom_bar(&mut self, config: &PciBarConfiguration, active: u32) -> Result<()> {
        let bar_idx = config.idx;
        let reg_idx = ROM_BAR_REG;

        if self.rom_bar_used {
            return Err(Error::RomBarInUse(bar_idx));
        }

        if !config.size.is_power_of_two() {
            return Err(Error::RomBarSizeInvalid(config.size));
        }

        if bar_idx != ROM_BAR_IDX {
            return Err(Error::RomBarInvalid(bar_idx));
        }

        let end_addr = config
            .addr
            .checked_add(config.size - 1)
            .ok_or(Error::RomBarAddressInvalid(config.addr, config.size))?;

        if end_addr > u64::from(u32::MAX) {
            return Err(Error::RomBarAddressInvalid(config.addr, config.size));
        }

        self.registers[reg_idx] = (config.addr as u32) | active;
        self.writable_bits[reg_idx] = ROM_BAR_ADDR_MASK;
        self.rom_bar_addr = self.registers[reg_idx];
        self.rom_bar_size =
            encode_32_bits_bar_size(config.size as u32).ok_or(Error::Encode32BarSize)?;
        self.rom_bar_used = true;

        // The ROM BAR is mapped (allocator/bus/guest-physical mapping) at
        // its initial address.
        self.rom_mapped_addr = Some(config.addr);

        Ok(())
    }

    /// Returns the address of the given BAR region.
    pub fn get_bar_addr(&self, bar_num: usize) -> u64 {
        let bar_idx = BAR0_REG + bar_num;

        let mut addr = u64::from(self.bars[bar_num].addr & self.writable_bits[bar_idx]);

        if let Some(bar_type) = self.bars[bar_num].r#type
            && bar_type == PciBarRegionType::Memory64BitRegion
        {
            addr |= u64::from(self.bars[bar_num + 1].addr) << 32;
        }

        addr
    }

    /// Configures the IRQ line and pin used by this device.
    pub fn set_irq(&mut self, line: u8, pin: PciInterruptPin) {
        // `pin` is 1-based in the pci config space.
        let pin_idx = (pin as u32) + 1;
        self.registers[INTERRUPT_LINE_PIN_REG] = (self.registers[INTERRUPT_LINE_PIN_REG]
            & 0xffff_0000)
            | (pin_idx << 8)
            | u32::from(line);
    }

    /// Adds the capability `cap_data` to the list of capabilities.
    /// `cap_data` should include the two-byte PCI capability header (type, next),
    /// but not populate it. Correct values will be generated automatically based
    /// on `cap_data.id()`.
    pub fn add_capability(&mut self, cap_data: &dyn PciCapability) -> Result<usize> {
        let total_len = cap_data.bytes().len();
        // Check that the length is valid.
        if cap_data.bytes().is_empty() {
            return Err(Error::CapabilityEmpty);
        }
        let (cap_offset, tail_offset) = match self.last_capability {
            Some((offset, len)) => (Self::next_dword(offset, len), offset + 1),
            None => (FIRST_CAPABILITY_OFFSET, CAPABILITY_LIST_HEAD_OFFSET),
        };
        let end_offset = cap_offset
            .checked_add(total_len)
            .ok_or(Error::CapabilitySpaceFull(total_len))?;
        if end_offset > CAPABILITY_MAX_OFFSET {
            return Err(Error::CapabilitySpaceFull(total_len));
        }
        self.registers[STATUS_REG] |= STATUS_REG_CAPABILITIES_USED_MASK;
        self.write_byte_internal(tail_offset, cap_offset as u8, false);
        self.write_byte_internal(cap_offset, cap_data.id() as u8, false);
        self.write_byte_internal(cap_offset + 1, 0, false); // Next pointer.
        for (i, byte) in cap_data.bytes().iter().enumerate() {
            self.write_byte_internal(cap_offset + i + 2, *byte, false);
        }
        self.last_capability = Some((cap_offset, total_len));

        match cap_data.id() {
            PciCapabilityId::MessageSignalledInterrupts => {
                self.writable_bits[cap_offset / 4] = MSI_CAPABILITY_REGISTER_MASK;
            }
            PciCapabilityId::MsiX => {
                self.msix_cap_reg_idx = Some(cap_offset / 4);
                self.writable_bits[self.msix_cap_reg_idx.unwrap()] = MSIX_CAPABILITY_REGISTER_MASK;
            }
            _ => {}
        }

        Ok(cap_offset)
    }

    // Find the next aligned offset after the one given.
    fn next_dword(offset: usize, len: usize) -> usize {
        let next = offset + len;
        (next + 3) & !3
    }

    pub fn write_config_register(
        &mut self,
        reg_idx: usize,
        offset: u64,
        data: &[u8],
    ) -> BarRelocation {
        if offset as usize + data.len() > 4 {
            return BarRelocation::default();
        }

        // Handle potential write to MSI-X message control register
        if let Some(msix_cap_reg_idx) = self.msix_cap_reg_idx
            && let Some(msix_config) = &self.msix_config
        {
            if msix_cap_reg_idx == reg_idx && offset == 2 && data.len() == 2 {
                msix_config
                    .lock()
                    .unwrap()
                    .set_msg_ctl(LittleEndian::read_u16(data));
            } else if msix_cap_reg_idx == reg_idx && offset == 0 && data.len() == 4 {
                msix_config
                    .lock()
                    .unwrap()
                    .set_msg_ctl((LittleEndian::read_u32(data) >> 16) as u16);
            }
        }

        // Capture the COMMAND register before the write so decode-enable
        // (IOSE/MSE 0->1) edges can be detected below.
        let command_before = self.registers[COMMAND_REG];

        match data.len() {
            1 => self.write_byte(reg_idx * 4 + offset as usize, data[0]),
            2 => self.write_word(
                reg_idx * 4 + offset as usize,
                u16::from(data[0]) | (u16::from(data[1]) << 8),
            ),
            4 => self.write_reg(reg_idx, LittleEndian::read_u32(data)),
            _ => (),
        }

        let mut reloc = BarRelocation::default();

        // Case 1: a BAR address was reprogrammed by this write.
        if let Some(params) = self.detect_bar_reprogramming(reg_idx, data) {
            let slot = params.bar_idx;

            if let Some(mapped) = self.mapped_addr(slot) {
                // Mapped: release the old BAR range eagerly and mark the
                // slot pending. Install is in the same plan if decode is on
                // (immediate move), else the decode-enable edge. mapped_addr
                // advances only on a successful install outcome.
                reloc.release.push(ReleaseParams {
                    bar_idx: slot,
                    base: mapped,
                    len: params.len,
                    region_type: params.region_type,
                });
                self.set_mapped_addr(slot, None);
                self.pending_relocation[slot] = Some(mapped);

                if self.decode_enabled(params.region_type) {
                    reloc.install.push(InstallParams {
                        bar_idx: slot,
                        // Released from `mapped`; confines the install to the
                        // PCI segment that address belongs to.
                        old_base: mapped,
                        new_base: params.new_base,
                        len: params.len,
                        region_type: params.region_type,
                    });
                }
            }
            // else: already released. Nothing to emit; the install target
            // is read from the live registers at drain time, and the
            // recorded released-from base stays intact.
        }

        // Case 2: commit each space whose decode bit rose 0->1 (shared with the
        // VFIO mirror). A non-COMMAND write sees no space enabling,
        // so this is a no-op.
        self.drain_command_decode_edge(command_before, &mut reloc);

        reloc
    }

    /// Drain the BAR unblocked by a COMMAND enable (IOSE/MSE 0->1) into
    /// `reloc`.
    pub(crate) fn drain_command_decode_edge(
        &mut self,
        command_before: u32,
        reloc: &mut BarRelocation,
    ) {
        let command_after = self.registers[COMMAND_REG];
        for space_mask in [COMMAND_REG_IO_SPACE_MASK, COMMAND_REG_MEMORY_SPACE_MASK] {
            if command_before & space_mask == 0 && command_after & space_mask != 0 {
                self.drain_pending_relocation(space_mask, reloc);
            }
        }

        if !reloc.is_empty() {
            info!("BAR relocation plan: {reloc:x?}");
        }
    }

    /// Emits the deferred installs for every pending BAR of the space whose
    /// enable bit is set. Only a successful install unpends a slot, so a failed one
    /// is retried at the guest's next decode-enable edge.
    fn drain_pending_relocation(&mut self, space_mask: u32, reloc: &mut BarRelocation) {
        let pending: Vec<usize> = (0..self.pending_relocation.len())
            .filter(|&slot| self.pending_relocation[slot].is_some())
            .collect();

        for slot in pending {
            let Some(region_type) = self.bar_region_type(slot) else {
                continue;
            };
            if Self::decode_mask(region_type) != space_mask {
                continue;
            }
            let len = self.bar_len(slot).unwrap_or(0);

            // A restored in-flight move (see `PciConfiguration::new`) left
            // the BAR mapped at its released-from address: replay the
            // release first so the whole move runs here exactly as it would
            // have run on the snapshot source.
            if let Some(mapped) = self.mapped_addr(slot) {
                reloc.release.push(ReleaseParams {
                    bar_idx: slot,
                    base: mapped,
                    len,
                    region_type,
                });
                self.set_mapped_addr(slot, None);
            }

            // The BAR belongs to the PCI segment whose MMIO window held the
            // address it was released from; carry that base so the install
            // confines the new range to the same segment window.
            let old_base = self.pending_relocation[slot].unwrap_or_else(|| self.bar_target(slot));

            reloc.install.push(InstallParams {
                bar_idx: slot,
                old_base,
                new_base: self.bar_target(slot),
                len,
                region_type,
            });
        }
    }

    /// Reconciles the mapping bookkeeping with an install status.
    /// `mapped_addr` advances ONLY here, on success. A failed install leaves
    /// the slot pending and unmapped -- nothing lies about being mapped --
    /// and it is re-emitted at the guest's next enabling the space.
    pub fn on_bar_relocation_status(&mut self, bar_idx: usize, status: BarRelocationStatus) {
        match status {
            BarRelocationStatus::Applied { base } => {
                self.set_mapped_addr(bar_idx, Some(base));
                self.pending_relocation[bar_idx] = None;
            }
            BarRelocationStatus::Pending => {
                // The slot was inserted into `pending_relocation` when its
                // release was emitted and stays there; nothing is mapped.
                if self.pending_relocation[bar_idx].is_none() {
                    warn!("BAR {bar_idx} install reported pending but no release was recorded");
                }
            }
        }
    }

    /// Returns true while the BAR is released awaiting install: its
    /// allocator range, bus range and guest-physical mappings are already
    /// torn down, so teardown paths (e.g. `free_bars`) must skip it.
    ///
    /// `pending_relocation` alone is not the right test: a restored
    /// in-flight move is pending AND mapped (the restore path materializes
    /// the BAR at its released-from base), so its ranges are live and
    /// teardown must NOT skip it -- hence the `mapped_addr` check.
    pub fn is_bar_released(&self, bar_idx: usize) -> bool {
        self.pending_relocation[bar_idx].is_some() && self.mapped_addr(bar_idx).is_none()
    }

    /// The COMMAND-register enabling bit for BAR's space: IOSE for IO
    /// BARs, MSE for 32/64-bit memory BARs and the expansion ROM.
    fn decode_mask(region_type: PciBarRegionType) -> u32 {
        match region_type {
            PciBarRegionType::IoRegion => COMMAND_REG_IO_SPACE_MASK,
            _ => COMMAND_REG_MEMORY_SPACE_MASK,
        }
    }

    fn decode_enabled(&self, region_type: PciBarRegionType) -> bool {
        self.registers[COMMAND_REG] & Self::decode_mask(region_type) != 0
    }

    /// Returns the currently-mapped address for a BAR slot (ROM_BAR_IDX for
    /// the ROM BAR), or `None` while the BAR is released.
    fn mapped_addr(&self, slot: usize) -> Option<u64> {
        if slot == ROM_BAR_IDX {
            self.rom_mapped_addr
        } else {
            self.mapped_addr[slot]
        }
    }

    fn set_mapped_addr(&mut self, slot: usize, addr: Option<u64>) {
        if slot == ROM_BAR_IDX {
            self.rom_mapped_addr = addr;
        } else {
            self.mapped_addr[slot] = addr;
        }
    }

    /// The install target for a BAR slot, assembled from the live
    /// guest-visible registers (masked by the writable bits, both dwords
    /// for a 64-bit BAR). Deliberately NOT the `bars[].addr` shadow: the
    /// shadow lags the registers when the guest writes the high dword of a
    /// 64-bit BAR first, or rewrites an address while the BAR is released.
    fn bar_target(&self, slot: usize) -> u64 {
        if slot == ROM_BAR_IDX {
            return u64::from(self.registers[ROM_BAR_REG] & self.writable_bits[ROM_BAR_REG]);
        }

        let reg_idx = BAR0_REG + slot;
        let mut addr = u64::from(self.registers[reg_idx] & self.writable_bits[reg_idx]);
        if self.bars[slot].r#type == Some(PciBarRegionType::Memory64BitRegion) {
            addr |= u64::from(self.registers[reg_idx + 1] & self.writable_bits[reg_idx + 1]) << 32;
        }

        addr
    }

    fn bar_len(&self, slot: usize) -> Option<u64> {
        if slot == ROM_BAR_IDX {
            return decode_32_bits_bar_size(self.rom_bar_size).map(u64::from);
        }

        match self.bars[slot].r#type? {
            PciBarRegionType::Memory64BitRegion => {
                decode_64_bits_bar_size(self.bars[slot + 1].size, self.bars[slot].size)
            }
            _ => decode_32_bits_bar_size(self.bars[slot].size).map(u64::from),
        }
    }

    fn bar_region_type(&self, slot: usize) -> Option<PciBarRegionType> {
        if slot == ROM_BAR_IDX {
            // The expansion ROM is a 32-bit memory region.
            return self
                .rom_bar_used
                .then_some(PciBarRegionType::Memory32BitRegion);
        }

        self.bars.get(slot)?.r#type
    }

    pub fn read_config_register(&self, reg_idx: usize) -> u32 {
        self.read_reg(reg_idx)
    }

    fn detect_bar_reprogramming(
        &mut self,
        reg_idx: usize,
        data: &[u8],
    ) -> Option<BarReprogramming> {
        if data.len() != 4 {
            return None;
        }

        let value = LittleEndian::read_u32(data);

        let mask = self.writable_bits[reg_idx];
        if (BAR0_REG..BAR0_REG + NUM_BAR_REGS).contains(&reg_idx) {
            // Ignore the case where the BAR size is being asked for.
            if value == 0xffff_ffff {
                return None;
            }

            let bar_idx = reg_idx - BAR0_REG;
            // Handle special case where the address being written is
            // different from the address initially provided. This is a
            // BAR reprogramming case which needs to be properly caught.
            if let Some(bar_type) = self.bars[bar_idx].r#type {
                // In case of 64 bits memory BAR, we don't do anything until
                // the upper BAR is modified, otherwise we would be moving the
                // BAR to a wrong location in memory.
                if bar_type == PciBarRegionType::Memory64BitRegion {
                    return None;
                }

                // Ignore the case where the value is unchanged.
                if (value & mask) == (self.bars[bar_idx].addr & mask) {
                    return None;
                }

                info!(
                    "Detected BAR reprogramming: (BAR {}) 0x{:x}->0x{:x}",
                    bar_idx, self.bars[bar_idx].addr, value
                );
                let new_base = u64::from(value & mask);
                let len = u64::from(
                    decode_32_bits_bar_size(self.bars[bar_idx].size)
                        .ok_or(Error::Decode32BarSize)
                        .unwrap(),
                );
                let region_type = bar_type;

                self.bars[bar_idx].addr = value;

                return Some(BarReprogramming {
                    bar_idx,
                    new_base,
                    len,
                    region_type,
                });
            } else if (bar_idx > 0)
                && ((self.registers[reg_idx - 1] & self.writable_bits[reg_idx - 1])
                    != (self.bars[bar_idx - 1].addr & self.writable_bits[reg_idx - 1])
                    || (value & mask) != (self.bars[bar_idx].addr & mask))
            {
                info!(
                    "Detected BAR reprogramming: (BAR {}) 0x{:x}->0x{:x}",
                    bar_idx, self.bars[bar_idx].addr, value
                );
                let new_base = (u64::from(value & mask) << 32)
                    | u64::from(self.registers[reg_idx - 1] & self.writable_bits[reg_idx - 1]);
                let len =
                    decode_64_bits_bar_size(self.bars[bar_idx].size, self.bars[bar_idx - 1].size)
                        .ok_or(Error::Decode64BarSize)
                        .unwrap();
                let region_type = PciBarRegionType::Memory64BitRegion;

                self.bars[bar_idx].addr = value;
                self.bars[bar_idx - 1].addr = self.registers[reg_idx - 1];

                // This branch fires on the HIGH-dword write; the BAR's
                // canonical slot (the one carrying r#type) is the
                // LOW/primary one.
                return Some(BarReprogramming {
                    bar_idx: bar_idx - 1,
                    new_base,
                    len,
                    region_type,
                });
            }
        } else if reg_idx == ROM_BAR_REG && (value & mask) != (self.rom_bar_addr & mask) {
            // Ignore the case where the BAR size is being asked for.
            if value & ROM_BAR_ADDR_MASK == ROM_BAR_ADDR_MASK {
                return None;
            }

            info!(
                "Detected ROM BAR reprogramming: (Expansion ROM BAR) 0x{:x}->0x{:x}",
                self.rom_bar_addr, value
            );
            let new_base = u64::from(value & mask);
            let len = u64::from(
                decode_32_bits_bar_size(self.rom_bar_size)
                    .ok_or(Error::Decode32BarSize)
                    .unwrap(),
            );
            let region_type = PciBarRegionType::Memory32BitRegion;

            self.rom_bar_addr = value;

            return Some(BarReprogramming {
                bar_idx: ROM_BAR_IDX,
                new_base,
                len,
                region_type,
            });
        }

        None
    }
}

impl Pausable for PciConfiguration {}

impl Snapshottable for PciConfiguration {
    fn id(&self) -> String {
        String::from(PCI_CONFIGURATION_ID)
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        Snapshot::new_from_state(&self.state())
    }
}

impl Default for PciBarConfiguration {
    fn default() -> Self {
        PciBarConfiguration {
            idx: 0,
            addr: 0,
            size: 0,
            region_type: PciBarRegionType::Memory64BitRegion,
            prefetchable: PciBarPrefetchable::NotPrefetchable,
        }
    }
}

impl PciBarConfiguration {
    pub fn new(
        idx: usize,
        size: u64,
        region_type: PciBarRegionType,
        prefetchable: PciBarPrefetchable,
    ) -> Self {
        PciBarConfiguration {
            idx,
            addr: 0,
            size,
            region_type,
            prefetchable,
        }
    }

    #[must_use]
    pub fn set_index(mut self, idx: usize) -> Self {
        self.idx = idx;
        self
    }

    #[must_use]
    pub fn set_address(mut self, addr: u64) -> Self {
        self.addr = addr;
        self
    }

    #[must_use]
    pub fn set_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    #[must_use]
    pub fn set_region_type(mut self, region_type: PciBarRegionType) -> Self {
        self.region_type = region_type;
        self
    }

    #[must_use]
    pub fn set_prefetchable(mut self, prefetchable: PciBarPrefetchable) -> Self {
        self.prefetchable = prefetchable;
        self
    }

    pub fn idx(&self) -> usize {
        self.idx
    }

    pub fn addr(&self) -> u64 {
        self.addr
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn region_type(&self) -> PciBarRegionType {
        self.region_type
    }

    pub fn prefetchable(&self) -> PciBarPrefetchable {
        self.prefetchable
    }
}

#[cfg(test)]
mod unit_tests {
    use vm_memory::ByteValued;

    use super::*;

    #[repr(C, packed)]
    #[derive(Clone, Copy, Default)]
    struct TestCap {
        len: u8,
        foo: u8,
    }

    // SAFETY: All members are simple numbers and any value is valid.
    unsafe impl ByteValued for TestCap {}

    impl PciCapability for TestCap {
        fn bytes(&self) -> &[u8] {
            self.as_slice()
        }

        fn id(&self) -> PciCapabilityId {
            PciCapabilityId::VendorSpecific
        }
    }

    #[test]
    fn add_capability() {
        let mut cfg = PciConfiguration::new(
            0x1234,
            0x5678,
            0x1,
            PciClassCode::MultimediaController,
            &PciMultimediaSubclass::AudioController,
            None,
            PciHeaderType::Device,
            0xABCD,
            0x2468,
            None,
            None,
        );

        // Add two capabilities with different contents.
        let cap1 = TestCap { len: 4, foo: 0xAA };
        let cap1_offset = cfg.add_capability(&cap1).unwrap();
        assert_eq!(cap1_offset % 4, 0);

        let cap2 = TestCap {
            len: 0x04,
            foo: 0x55,
        };
        let cap2_offset = cfg.add_capability(&cap2).unwrap();
        assert_eq!(cap2_offset % 4, 0);

        // The capability list head should be pointing to cap1.
        let cap_ptr = cfg.read_reg(CAPABILITY_LIST_HEAD_OFFSET / 4) & 0xFF;
        assert_eq!(cap1_offset, cap_ptr as usize);

        // Verify the contents of the capabilities.
        let cap1_data = cfg.read_reg(cap1_offset / 4);
        assert_eq!(cap1_data & 0xFF, 0x09); // capability ID
        assert_eq!((cap1_data >> 8) & 0xFF, cap2_offset as u32); // next capability pointer
        assert_eq!((cap1_data >> 16) & 0xFF, 0x04); // cap1.len
        assert_eq!((cap1_data >> 24) & 0xFF, 0xAA); // cap1.foo

        let cap2_data = cfg.read_reg(cap2_offset / 4);
        assert_eq!(cap2_data & 0xFF, 0x09); // capability ID
        assert_eq!((cap2_data >> 8) & 0xFF, 0x00); // next capability pointer
        assert_eq!((cap2_data >> 16) & 0xFF, 0x04); // cap2.len
        assert_eq!((cap2_data >> 24) & 0xFF, 0x55); // cap2.foo
    }

    #[derive(Copy, Clone)]
    enum TestPi {
        Test = 0x5a,
    }

    impl PciProgrammingInterface for TestPi {
        fn get_register_value(&self) -> u8 {
            *self as u8
        }
    }

    #[test]
    fn class_code() {
        let cfg = PciConfiguration::new(
            0x1234,
            0x5678,
            0x1,
            PciClassCode::MultimediaController,
            &PciMultimediaSubclass::AudioController,
            Some(&TestPi::Test),
            PciHeaderType::Device,
            0xABCD,
            0x2468,
            None,
            None,
        );

        let class_reg = cfg.read_reg(2);
        let class_code = (class_reg >> 24) & 0xFF;
        let subclass = (class_reg >> 16) & 0xFF;
        let prog_if = (class_reg >> 8) & 0xFF;
        assert_eq!(class_code, 0x04);
        assert_eq!(subclass, 0x01);
        assert_eq!(prog_if, 0x5a);
    }
}
