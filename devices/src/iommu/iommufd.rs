// Copyright © 2026 Cloud Hypervisor Contributors
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use iommufd_bindings::iommufd::{
    iommu_hw_info, iommu_hw_info_arm_smmuv3, iommu_hwpt_arm_smmuv3,
    iommu_veventq_flag_IOMMU_VEVENTQ_FLAG_LOST_EVENTS, iommu_viommu_arm_smmuv3_invalidate,
    iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_ATS_NOT_SUPPORTED,
    iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_PASID_EXEC,
    iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_PASID_PRIV,
};
use iommufd_ioctls::{
    AttachHwpt, IommufdError, IommufdHwInfoData, IommufdHwptData, IommufdInvalidateData,
    IommufdVDevice, IommufdVEvent, IommufdVEventData, IommufdVEventQ, IommufdVIommu,
    IommufdViommuData,
};
use log::{debug, error, warn};
use pci::{PasidCap, PciBdf};
use thiserror::Error;
use vm_memory::bitmap::AtomicBitmap;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

use crate::iommu::{
    Error as IommuError, HwInfo, Invalidation, PhysicalIommu, Smmuv3AcpiInfo, TableEntry,
};
use crate::smmuv3::{Error as Smmuv3Error, EventRecord, IDR0_COHACC, Smmuv3, Smmuv3Interrupts};

const VEVENTQ_DEPTH: u32 = 64;

// The STE fields the kernel lets userspace set for a nested stage 1
// (STRTAB_STE_{0,1}_NESTING_ALLOWED, arm-smmu-v3.h). It refuses an STE
// with any other bit set (EIO), and a Linux guest does set others: SHCFG
// whenever S1DSS is bypass, as for a PASID endpoint whose default domain
// is identity. The kernel derives the rest from its own stage 2.
//   word 0: V [0], Config [3:1], S1Fmt [5:4], S1ContextPtr [51:6],
//           S1CDMax [63:59]
//   word 1: S1DSS [1:0], S1CIR [3:2], S1COR [5:4], S1CSH [7:6],
//           S1STALLD [27], EATS [29:28]
#[cfg(target_arch = "aarch64")]
const STE0_NESTING_ALLOWED: u64 = 0xf80f_ffff_ffff_ffff;
#[cfg(target_arch = "aarch64")]
const STE1_NESTING_ALLOWED: u64 = 0x0000_0000_3800_00ff;

/// The two STE words the kernel takes for a nested stage 1, cut down to
/// the fields it accepts.
#[cfg(target_arch = "aarch64")]
fn nested_ste(words: &[u64; 8]) -> [u64; 2] {
    [
        words[0] & STE0_NESTING_ALLOWED,
        words[1] & STE1_NESTING_ALLOWED,
    ]
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Failed to query host IOMMU information")]
    QueryHwInfo(#[source] IommufdError),
    #[error("Failed to create vDevice")]
    CreateVdevice(#[source] IommufdError),
    #[error("Failed to attach device to the bypass HWPT")]
    AttachBypass(#[source] IommufdError),
    #[error("Failed to install stage 1 HWPT")]
    InstallNestedStage1(#[source] io::Error),
    #[error("Failed to uninstall stage 1 HWPT")]
    UninstallStage1(#[source] IommufdError),
    #[error("Failed to invalidate vIOMMU caches")]
    Invalidate(#[source] IommufdError),
    #[error("Failed to allocate vEVENTQ")]
    AllocateVeventq(#[source] IommufdError),
    #[error("Failed to create fault forwarder eventfd")]
    CreateEventFd(#[source] io::Error),
    #[error("Failed to spawn fault forwarder thread")]
    SpawnFaultForwarder(#[source] io::Error),
    #[error("Failed to initialize the emulated SMMUv3 from host information")]
    DeviceInit(#[source] Smmuv3Error),
}

impl From<Error> for IommuError {
    fn from(e: Error) -> Self {
        IommuError::Backend(io::Error::other(e))
    }
}

/// Device attached to the vIOMMU.
struct Endpoint {
    bdf: PciBdf,
    device: Arc<dyn AttachHwpt>,
    /// Held for its lifetime: dropping it destroys the vDevice.
    _vdevice: IommufdVDevice,
    /// The iommufd device id the vDevice was allocated for; a nested
    /// stage-1 HWPT is allocated against it.
    dev_id: u32,
    /// The stage-1 this endpoint is attached to. Owned here rather than by
    /// `IommufdVDevice`, whose install/uninstall verbs can only replace a
    /// stage-1 by parking the endpoint on the abort HWPT first.
    s1: S1State,
}

/// The stage-1 bookkeeping of one endpoint: the nested HWPT it is attached
/// to (if any), and the EATS bit and masked STE of that HWPT. `ste` is
/// `Some` exactly while a guest stage-1 is attached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct S1State {
    hwpt_id: Option<u32>,
    eats: bool,
    ste: Option<[u64; 2]>,
}

/// What to do with an endpoint whose stage-1 install failed, decided by
/// whether a guest stage-1 was attached when the install began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstallFailureAction {
    /// No stage-1 was attached: the endpoint is on the bypass HWPT while
    /// the guest believes stage-1 is enforced, and could DMA anywhere in
    /// guest RAM. Fail closed on the abort HWPT.
    ParkOnAbort,
    /// A guest stage-1 was attached and, by the kernel's replace contract
    /// (a failed `iommufd_device_replace` changes nothing), still is. Keep
    /// it: a stale guest translation is bounded by the stage-2 parent,
    /// whereas aborting the stream terminates its in-flight DMA, which
    /// GB300-class GPUs escalate to an uncontained Xid. Upstream QEMU
    /// behaves the same.
    KeepInstalled,
}

fn install_failure_action(guest_s1_attached: bool) -> InstallFailureAction {
    if guest_s1_attached {
        InstallFailureAction::KeepInstalled
    } else {
        InstallFailureAction::ParkOnAbort
    }
}

/// STE.EATS is word 1 bits [29:28]; non-zero means the endpoint may hold
/// an ATC.
fn ste_eats(ste: &[u64; 2]) -> bool {
    ((ste[1] >> 28) & 0b11) != 0
}

/// A full stage-1 TLB invalidation: CMD_TLBI_NH_ALL (opcode 0x10), every
/// other field zero; the kernel inserts the VMID.
const CMD_TLBI_NH_ALL: [u64; 2] = [0x10, 0];

/// A whole-ATC invalidation of one stream: CMD_ATC_INV (opcode 0x40) with the
/// virtual StreamID in word 0 bits [63:32] (the kernel rewrites it to the
/// physical one), SSV clear for every substream, and Size 52 in word 1
/// bits [5:0], the architected invalidate-all span (Linux ATC_INV_SIZE_ALL).
fn cmd_atc_inv_all(sid: u32) -> [u64; 2] {
    [0x40 | (u64::from(sid) << 32), 52]
}

fn hwpt_name(id: Option<u32>) -> String {
    id.map_or_else(|| "none".to_string(), |id| format!("S1 HWPT {id}"))
}

/// The vIOMMU operations the stage-1 install needs; a trait so the
/// install policy can be tested without an iommufd.
trait S1HwptOps {
    fn alloc_s1_hwpt(&self, dev_id: u32, ste: [u64; 2]) -> io::Result<u32>;
    fn destroy_hwpt(&self, id: u32) -> io::Result<()>;
    fn park_on_abort(&self, device: &dyn AttachHwpt) -> io::Result<()>;
}

impl S1HwptOps for IommufdVIommu {
    fn alloc_s1_hwpt(&self, dev_id: u32, ste: [u64; 2]) -> io::Result<u32> {
        let data = IommufdHwptData::Smmuv3(iommu_hwpt_arm_smmuv3 { ste });
        IommufdVIommu::alloc_s1_hwpt(self, dev_id, &data).map_err(io::Error::other)
    }

    fn destroy_hwpt(&self, id: u32) -> io::Result<()> {
        self.iommufd()
            .destroy_iommu_object(id)
            .map_err(io::Error::other)
    }

    fn park_on_abort(&self, device: &dyn AttachHwpt) -> io::Result<()> {
        self.attach_abort(device).map_err(io::Error::other)
    }
}

/// Install `ste` as the endpoint's stage-1 in the shape upstream QEMU's
/// smmuv3_accel_install_ste() ships:
///
///   stage   allocate the new HWPT while the current attachment, bypass or
///           the previous stage-1, stays in place;
///   attach  one attach to the staged HWPT, which the kernel services as
///           iommufd_device_replace, so no abort HWPT sits between the old
///           translation and the new one and in-flight DMA is never aborted;
///   commit  destroy the previous HWPT, now that nothing is attached to it.
///
/// The cache is written at exactly one point, after the attach succeeds, so
/// a failed destroy-of-old leaves it describing the HWPT the endpoint is on;
/// that HWPT leaks until the iommufd context closes, with one warning and no
/// park. On a failed stage or attach, `install_failure_action` decides, and
/// the install reports failure.
///
/// The log lines are counted by the hardware gates by substring
/// ("installed nested", "staged S1 HWPT N (previous:", and the failure
/// wording); do not reword them without the counters.
fn install_s1(
    ops: &dyn S1HwptOps,
    device: &dyn AttachHwpt,
    sid: u32,
    dev_id: u32,
    ste: [u64; 2],
    st: &mut S1State,
) -> io::Result<()> {
    let failed =
        |ops: &dyn S1HwptOps, st: &mut S1State| match install_failure_action(st.ste.is_some()) {
            InstallFailureAction::KeepInstalled => {
                debug!("SMMUv3 accel: SID {sid:#x} keeps its current stage-1");
            }
            InstallFailureAction::ParkOnAbort => {
                if let Err(e) = ops.park_on_abort(device) {
                    warn!("SMMUv3 accel: SID {sid:#x} park on the abort HWPT failed: {e}");
                }
                st.eats = false;
                st.ste = None;
            }
        };
    let previous = hwpt_name(st.hwpt_id);

    // Stage.
    let staged = match ops.alloc_s1_hwpt(dev_id, ste) {
        Ok(id) => id,
        Err(e) => {
            warn!("SMMUv3 accel: install nested S1 for SID {sid:#x} failed: {e}");
            failed(ops, st);
            return Err(e);
        }
    };
    debug!("SMMUv3 accel: SID {sid:#x} staged S1 HWPT {staged} (previous: {previous})");

    // Attach: a replace, from bypass or from the previous stage-1.
    if let Err(e) = device.attach_hwpt(staged) {
        warn!("SMMUv3 accel: SID {sid:#x} attach to S1 HWPT {staged} failed: {e}");
        // Nothing is attached to the staged HWPT; drop it.
        if let Err(e) = ops.destroy_hwpt(staged) {
            warn!("SMMUv3 accel: SID {sid:#x} destroy staged S1 HWPT {staged} failed: {e}");
        }
        failed(ops, st);
        return Err(e);
    }

    let old = st.hwpt_id.replace(staged);
    st.eats = ste_eats(&ste);
    st.ste = Some(ste);

    // Commit.
    match old.map(|old| (old, ops.destroy_hwpt(old))) {
        Some((old, Err(e))) => warn!(
            "SMMUv3 accel: installed nested S1 HWPT {staged} for SID {sid:#x} but the previous \
             S1 HWPT {old} could not be destroyed ({e}); leaked until the iommufd context closes"
        ),
        _ => debug!(
            "SMMUv3 accel: installed nested S1 HWPT {staged} for SID {sid:#x} \
             (previous: {previous})"
        ),
    }

    Ok(())
}

/// Iommufd backend of an emulated IOMMU.
pub struct IommufdIommu {
    viommu: Arc<IommufdVIommu>,
    hw_info_data: IommufdHwInfoData,
    ats_supported: bool,
    endpoints: Mutex<BTreeMap<u32, Endpoint>>,
}

impl IommufdIommu {
    pub fn new(viommu: Arc<IommufdVIommu>, dev_id: u32) -> Result<Self, Error> {
        let (hw_info, hw_info_data) = Self::query_hw_info(&viommu, dev_id)?;

        Ok(Self {
            viommu,
            hw_info_data,
            ats_supported: hw_info.out_capabilities
                & u64::from(iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_ATS_NOT_SUPPORTED)
                == 0,
            endpoints: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn allocate_veventq(&self) -> Result<IommufdVEventQ, Error> {
        self.viommu
            .allocate_veventq(VEVENTQ_DEPTH)
            .map_err(Error::AllocateVeventq)
    }

    pub fn register_endpoint(
        &self,
        virt_id: u32,
        bdf: PciBdf,
        device: Arc<dyn AttachHwpt>,
        dev_id: u32,
    ) -> Result<(), Error> {
        let vdevice = IommufdVDevice::new(Arc::clone(&self.viommu), dev_id, u64::from(virt_id))
            .map_err(Error::CreateVdevice)?;
        self.viommu
            .attach_bypass(&*device)
            .map_err(Error::AttachBypass)?;

        self.endpoints.lock().unwrap().insert(
            virt_id,
            Endpoint {
                bdf,
                device,
                _vdevice: vdevice,
                dev_id,
                s1: S1State::default(),
            },
        );

        Ok(())
    }

    pub fn pasid_cap(&self, dev_id: u32) -> Result<Option<PasidCap>, Error> {
        let (hw_info, _) = Self::query_hw_info(&self.viommu, dev_id)?;
        Ok((hw_info.out_max_pasid_log2 != 0).then(|| {
            PasidCap::new(
                hw_info.out_max_pasid_log2,
                hw_info.out_capabilities
                    & u64::from(iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_PASID_EXEC)
                    != 0,
                hw_info.out_capabilities
                    & u64::from(iommufd_hw_capabilities_IOMMU_HW_CAP_PCI_PASID_PRIV)
                    != 0,
            )
        }))
    }

    pub fn attached_bdfs(&self) -> Vec<PciBdf> {
        self.endpoints
            .lock()
            .unwrap()
            .values()
            .map(|endpoint| endpoint.bdf)
            .collect()
    }

    fn query_hw_info(
        viommu: &IommufdVIommu,
        dev_id: u32,
    ) -> Result<(iommu_hw_info, IommufdHwInfoData), Error> {
        // The buffer to fill depends on the IOMMU the host reports.
        let mut hw_info_data = match viommu.data() {
            IommufdViommuData::Smmuv3 | IommufdViommuData::Tegra241Cmdqv { .. } => {
                IommufdHwInfoData::Smmuv3(iommu_hw_info_arm_smmuv3::default())
            }
        };
        let hw_info = viommu
            .iommufd()
            .device_hw_info(dev_id, &mut hw_info_data)
            .map_err(Error::QueryHwInfo)?;

        Ok((hw_info, hw_info_data))
    }

    /// Forward one raw command; whether the kernel consumed it.
    fn forward_one(&self, cmd: [u64; 2]) -> bool {
        let mut data = IommufdInvalidateData::Smmuv3(iommu_viommu_arm_smmuv3_invalidate { cmd });
        matches!(self.viommu.invalidate(&mut data), Ok(true))
    }

    /// Recover from an invalidation the host did not apply. The kernel's
    /// partial-success protocol (entry_num) over-reports on convert errors,
    /// counting commands converted but never submitted, so the only safe
    /// recovery is a full re-invalidate, never a resume. A full stage-1 TLB
    /// invalidation cannot clear a device's ATC, so every endpoint whose
    /// stage-1 enabled ATS also gets a whole-ATC invalidation, as the kernel
    /// guards its own ATC invalidations by ATS being enabled. One attempt
    /// each, no retry.
    fn resync(&self) {
        warn!("SMMUv3 accel: resyncing with TLBI_NH_ALL");
        if !self.forward_one(CMD_TLBI_NH_ALL) {
            warn!("SMMUv3 accel: TLBI_NH_ALL resync forward failed; giving up");
        }
        if !self.ats_supported {
            return;
        }
        for (&sid, endpoint) in self.endpoints.lock().unwrap().iter() {
            if endpoint.s1.eats && !self.forward_one(cmd_atc_inv_all(sid)) {
                warn!("SMMUv3 accel: ATC_INV resync forward for SID {sid:#x} failed; giving up");
            }
        }
    }

    /// Move the endpoint to the bypass (`abort == false`) or abort HWPT and
    /// destroy its stage-1. The park comes first: an HWPT cannot be
    /// destroyed while a device is attached to it.
    fn uninstall_s1(&self, device_id: u32, abort: bool) -> Result<(), IommuError> {
        let mut endpoints = self.endpoints.lock().unwrap();
        let Some(endpoint) = endpoints.get_mut(&device_id) else {
            return Ok(());
        };
        if abort {
            self.viommu.attach_abort(&*endpoint.device)
        } else {
            self.viommu.attach_bypass(&*endpoint.device)
        }
        .map_err(Error::UninstallStage1)?;
        endpoint.s1.eats = false;
        endpoint.s1.ste = None;

        if let Some(id) = endpoint.s1.hwpt_id {
            match self.viommu.iommufd().destroy_iommu_object(id) {
                Ok(()) => {
                    endpoint.s1.hwpt_id = None;
                    // Counted by the hardware gates as "uninstall done".
                    debug!(
                        "SMMUv3 accel: uninstall done: nested S1 HWPT {id} for SID \
                         {device_id:#x} (now on the {} HWPT)",
                        if abort { "abort" } else { "bypass" }
                    );
                }
                // The id stays, so the next install's commit reclaims it.
                Err(e) => {
                    warn!("SMMUv3 accel: SID {device_id:#x} destroy S1 HWPT {id} failed: {e}");
                }
            }
        }

        Ok(())
    }
}

impl Drop for IommufdIommu {
    fn drop(&mut self) {
        let endpoints = self.endpoints.get_mut().unwrap();
        for endpoint in endpoints.values() {
            if let Err(e) = endpoint.device.detach_hwpt() {
                error!(
                    "Failed to detach device {} from the vIOMMU: {e}",
                    endpoint.bdf
                );
            }
        }
        // The stage-1 HWPTs are this backend's, not the vDevices': destroy
        // them now that nothing is attached to them.
        for endpoint in endpoints.values_mut() {
            if let Some(id) = endpoint.s1.hwpt_id.take()
                && let Err(e) = self.viommu.iommufd().destroy_iommu_object(id)
            {
                error!(
                    "Failed to destroy S1 HWPT {id} of device {}: {e}",
                    endpoint.bdf
                );
            }
        }
    }
}

impl PhysicalIommu for IommufdIommu {
    fn hw_info(&self) -> Result<HwInfo, IommuError> {
        match self.hw_info_data {
            #[cfg(target_arch = "aarch64")]
            IommufdHwInfoData::Smmuv3(info) => Ok(HwInfo::Smmuv3 {
                idr: info.idr,
                ats_supported: self.ats_supported,
            }),
        }
    }

    fn install_table_entry(&self, device_id: u32, entry: TableEntry) -> Result<(), IommuError> {
        let ste = match entry {
            #[cfg(target_arch = "aarch64")]
            TableEntry::Smmuv3Ste(words) => nested_ste(&words),
        };

        let mut endpoints = self.endpoints.lock().unwrap();
        let Some(endpoint) = endpoints.get_mut(&device_id) else {
            return Ok(());
        };
        let Endpoint {
            device, dev_id, s1, ..
        } = endpoint;
        install_s1(&*self.viommu, &**device, device_id, *dev_id, ste, s1)
            .map_err(|e| Error::InstallNestedStage1(e).into())
    }

    fn set_passthrough(&self, device_id: u32) -> Result<(), IommuError> {
        self.uninstall_s1(device_id, false)
    }

    fn set_blocking(&self, device_id: u32) -> Result<(), IommuError> {
        self.uninstall_s1(device_id, true)
    }

    fn invalidate(&self, invalidation: Invalidation) -> Result<(), IommuError> {
        let mut data = match invalidation {
            #[cfg(target_arch = "aarch64")]
            Invalidation::Smmuv3Cmd(cmd) => {
                IommufdInvalidateData::Smmuv3(iommu_viommu_arm_smmuv3_invalidate { cmd })
            }
        };
        match self.viommu.invalidate(&mut data) {
            Ok(true) => Ok(()),
            Ok(false) => {
                warn!("SMMUv3 accel: invalidation forwarded but no entry consumed");
                self.resync();
                Ok(())
            }
            Err(e) => {
                warn!("SMMUv3 accel: invalidation forward failed: {e}");
                self.resync();
                Err(Error::Invalidate(e).into())
            }
        }
    }
}

/// Emulated SMMUv3 backed by iommufd.
#[cfg(target_arch = "aarch64")]
pub struct Smmuv3Iommufd {
    device: Arc<Mutex<Smmuv3>>,
    backend: Arc<IommufdIommu>,
    acpi_info: Smmuv3AcpiInfo,
    _fault_forwarder: FaultForwarder,
}

#[cfg(target_arch = "aarch64")]
impl Smmuv3Iommufd {
    pub fn new(
        id: String,
        mem: GuestMemoryAtomic<GuestMemoryMmap<AtomicBitmap>>,
        interrupts: Smmuv3Interrupts,
        acpi_info: Smmuv3AcpiInfo,
        viommu: Arc<IommufdVIommu>,
        dev_id: u32,
    ) -> Result<Self, Error> {
        let backend = Arc::new(IommufdIommu::new(viommu, dev_id)?);
        let IommufdHwInfoData::Smmuv3(host_info) = backend.hw_info_data;
        let acpi_info = Smmuv3AcpiInfo {
            coherent: host_info.idr[0] & IDR0_COHACC != 0,
            ats_supported: backend.ats_supported,
            ..acpi_info
        };
        let mut device = Smmuv3::new(
            id,
            mem,
            interrupts,
            Arc::clone(&backend) as Arc<dyn PhysicalIommu>,
            None,
        );
        device.initialize().map_err(Error::DeviceInit)?;
        let device = Arc::new(Mutex::new(device));
        let fault_forwarder =
            FaultForwarder::new(backend.allocate_veventq()?, Arc::clone(&device))?;

        Ok(Self {
            device,
            backend,
            acpi_info,
            _fault_forwarder: fault_forwarder,
        })
    }

    /// Stream ID of a device encoded similarly to the ITS Device ID in the
    /// IORT table. There are 256 identifiers per PCI segment with a single
    /// bus per segment.
    pub fn stream_id(bdf: PciBdf) -> u32 {
        256 * u32::from(bdf.segment()) + (u32::from(bdf) & 0xff)
    }

    pub fn device(&self) -> &Arc<Mutex<Smmuv3>> {
        &self.device
    }

    pub fn backend(&self) -> &Arc<IommufdIommu> {
        &self.backend
    }

    pub fn acpi_info(&self) -> Smmuv3AcpiInfo {
        Smmuv3AcpiInfo {
            attached_bdfs: self.backend.attached_bdfs(),
            ..self.acpi_info.clone()
        }
    }
}

/// Thread forwarding vEVENTQ records to the emulated SMMUv3.
struct FaultForwarder {
    kill: EventFd,
    handle: Option<JoinHandle<()>>,
}

impl FaultForwarder {
    fn new(mut veventq: IommufdVEventQ, device: Arc<Mutex<Smmuv3>>) -> Result<Self, Error> {
        let kill = EventFd::new(EFD_NONBLOCK).map_err(Error::CreateEventFd)?;
        let kill_reader = kill.try_clone().map_err(Error::CreateEventFd)?;

        let handle = thread::Builder::new()
            .name("smmuv3_veventq".to_string())
            .spawn(move || Self::epoll_event_queue(&mut veventq, &device, &kill_reader))
            .map_err(Error::SpawnFaultForwarder)?;

        Ok(FaultForwarder {
            kill,
            handle: Some(handle),
        })
    }

    fn epoll_event_queue(veventq: &mut IommufdVEventQ, device: &Mutex<Smmuv3>, kill: &EventFd) {
        const VEVENTQ_TOKEN: u64 = 0;
        const KILL_TOKEN: u64 = 1;

        let epoll = match Epoll::new() {
            Ok(epoll) => epoll,
            Err(e) => {
                error!("SMMUv3 vEVENTQ reader failed to create epoll: {e}");
                return;
            }
        };
        for (fd, token) in [
            (veventq.as_raw_fd(), VEVENTQ_TOKEN),
            (kill.as_raw_fd(), KILL_TOKEN),
        ] {
            if let Err(e) = epoll.ctl(
                ControlOperation::Add,
                fd,
                EpollEvent::new(EventSet::IN, token),
            ) {
                error!("SMMUv3 vEVENTQ reader failed to register fd {fd}: {e}");
                return;
            }
        }

        let mut events = [EpollEvent::default(); 2];
        loop {
            let count = match epoll.wait(-1, &mut events) {
                Ok(count) => count,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    error!("SMMUv3 vEVENTQ reader epoll wait failed: {e}");
                    return;
                }
            };

            for event in events.iter().take(count) {
                match event.data() {
                    KILL_TOKEN => return,
                    VEVENTQ_TOKEN => match veventq.read_events() {
                        Ok(records) => {
                            let mut device = device.lock().unwrap();
                            for record in records {
                                Self::forward_record(&mut device, record);
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                        Err(e) => warn!("SMMUv3 vEVENTQ read failed: {e}"),
                    },
                    _ => unreachable!(),
                }
            }
        }
    }

    fn forward_record(device: &mut Smmuv3, record: IommufdVEvent) {
        if record.lost > 0 {
            warn!(
                "SMMUv3 vEVENTQ lost {} events before sequence {}",
                record.lost, record.header.sequence
            );
            device.set_event_overflow();
        }

        match record.data {
            Some(IommufdVEventData::Smmuv3(evt)) => {
                if let Err(e) = device.push_event(&EventRecord(evt.evt)) {
                    warn!("SMMUv3 failed to push a guest event: {e}");
                }
            }
            Some(IommufdVEventData::Tegra241Cmdqv(_)) => {
                warn!("SMMUv3 vEVENTQ yielded an unexpected CMDQV record");
            }
            // Header without a record.
            None => {
                if record.header.flags & iommu_veventq_flag_IOMMU_VEVENTQ_FLAG_LOST_EVENTS == 0 {
                    warn!("SMMUv3 vEVENTQ yielded a header with no record");
                } else {
                    warn!("SMMUv3 vEVENTQ lost events at the tail");
                }
                device.set_event_overflow();
            }
        }
    }
}

impl Drop for FaultForwarder {
    fn drop(&mut self) {
        if let Err(e) = self.kill.write(1) {
            error!("SMMUv3 failed to signal vEVENTQ reader shutdown: {e}");
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use std::mem;

    use super::*;

    /// Bits [hi:lo] set, as the kernel's GENMASK_ULL.
    fn genmask(hi: u32, lo: u32) -> u64 {
        (u64::MAX >> (63 - hi)) & (u64::MAX << lo)
    }

    #[test]
    fn test_nested_ste_mask_matches_the_kernel() {
        let ste0 = 1 | genmask(3, 1) | genmask(5, 4) | genmask(51, 6) | genmask(63, 59);
        let ste1 = genmask(1, 0)
            | genmask(3, 2)
            | genmask(5, 4)
            | genmask(7, 6)
            | (1 << 27)
            | genmask(29, 28);
        assert_eq!(STE0_NESTING_ALLOWED, ste0);
        assert_eq!(STE1_NESTING_ALLOWED, ste1);
    }

    #[test]
    fn test_nested_ste_drops_fields_the_kernel_refuses() {
        // SHCFG [45:44] = incoming, as Linux writes with S1DSS bypass;
        // STRW [31:30] and the word 2-7 stage-2 fields are the kernel's.
        let shcfg_incoming = 1 << 44;
        let strw = 0b10 << 30;
        let words = [
            u64::MAX,
            0b01 | (1 << 28) | shcfg_incoming | strw,
            u64::MAX,
            u64::MAX,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(nested_ste(&words), [STE0_NESTING_ALLOWED, 0b01 | (1 << 28)]);
    }

    /// Every operation the install path performs, in order.
    #[derive(Default)]
    struct Log(Mutex<Vec<String>>);

    impl Log {
        fn push(&self, e: String) {
            self.0.lock().unwrap().push(e);
        }
        fn take(&self) -> Vec<String> {
            mem::take(&mut *self.0.lock().unwrap())
        }
    }

    const ABORT: u32 = 77;
    const NEW: u32 = 50;
    const OLD: u32 = 9;
    const SID: u32 = 0x20;
    const STE_EATS: [u64; 2] = [0x5, 1 << 28];
    const STE_NO_EATS: [u64; 2] = [0x5, 0];

    struct FakeOps<'a> {
        log: &'a Log,
        alloc_ok: bool,
        destroy_ok: bool,
    }

    impl S1HwptOps for FakeOps<'_> {
        fn alloc_s1_hwpt(&self, _dev_id: u32, _ste: [u64; 2]) -> io::Result<u32> {
            self.log.push("alloc".into());
            if self.alloc_ok {
                Ok(NEW)
            } else {
                Err(io::Error::other("alloc"))
            }
        }
        fn destroy_hwpt(&self, id: u32) -> io::Result<()> {
            self.log.push(format!("destroy {id}"));
            if self.destroy_ok {
                Ok(())
            } else {
                Err(io::Error::other("destroy"))
            }
        }
        fn park_on_abort(&self, device: &dyn AttachHwpt) -> io::Result<()> {
            device.attach_hwpt(ABORT)
        }
    }

    struct FakeDevice<'a> {
        log: &'a Log,
        attach_ok: bool,
    }

    impl AttachHwpt for FakeDevice<'_> {
        fn attach_hwpt(&self, pt_id: u32) -> io::Result<()> {
            self.log.push(format!("attach {pt_id}"));
            // Parking on abort always succeeds; only the staged attach can
            // be made to fail.
            if self.attach_ok || pt_id == ABORT {
                Ok(())
            } else {
                Err(io::Error::other("attach"))
            }
        }
        fn detach_hwpt(&self) -> io::Result<()> {
            Ok(())
        }
    }

    fn on_guest_s1() -> S1State {
        S1State {
            hwpt_id: Some(OLD),
            eats: false,
            ste: Some(STE_NO_EATS),
        }
    }

    fn run(
        start: S1State,
        alloc_ok: bool,
        attach_ok: bool,
        destroy_ok: bool,
    ) -> (bool, S1State, Vec<String>) {
        let log = Log::default();
        let ops = FakeOps {
            log: &log,
            alloc_ok,
            destroy_ok,
        };
        let device = FakeDevice {
            log: &log,
            attach_ok,
        };
        let mut st = start;
        let ok = install_s1(&ops, &device, SID, 2, STE_EATS, &mut st).is_ok();
        (ok, st, log.take())
    }

    #[test]
    fn test_install_from_bypass_is_one_replace_with_no_park() {
        let (ok, st, log) = run(S1State::default(), true, true, true);
        assert!(ok);
        assert_eq!(log, ["alloc", "attach 50"]);
        assert_eq!(
            st,
            S1State {
                hwpt_id: Some(NEW),
                eats: true,
                ste: Some(STE_EATS)
            }
        );
    }

    #[test]
    fn test_s1_to_s1_replace_attaches_before_destroying_and_never_parks() {
        // The GB300 abort window: the previous stage-1 stays attached until
        // the new one replaces it, and only then is destroyed. No
        // "attach 77" (abort) may appear.
        let (ok, st, log) = run(on_guest_s1(), true, true, true);
        assert!(ok);
        assert_eq!(log, ["alloc", "attach 50", "destroy 9"]);
        assert_eq!(st.hwpt_id, Some(NEW));
        assert_eq!(st.ste, Some(STE_EATS));
        assert!(st.eats);
    }

    #[test]
    fn test_failed_commit_leaves_the_cache_on_the_new_stage1() {
        let (ok, st, log) = run(on_guest_s1(), true, true, false);
        assert!(ok, "the endpoint is on the new stage-1");
        assert_eq!(log, ["alloc", "attach 50", "destroy 9"]);
        assert_eq!(st.hwpt_id, Some(NEW));
        assert_eq!(st.ste, Some(STE_EATS));
    }

    #[test]
    fn test_failed_stage_keeps_an_attached_stage1() {
        let (ok, st, log) = run(on_guest_s1(), false, true, true);
        assert!(!ok);
        assert_eq!(log, ["alloc"], "no park, no attach");
        assert_eq!(st, on_guest_s1(), "cache still describes the attached S1");
    }

    #[test]
    fn test_failed_stage_without_a_stage1_parks_on_abort() {
        let (ok, st, log) = run(S1State::default(), false, true, true);
        assert!(!ok);
        assert_eq!(log, ["alloc", "attach 77"]);
        assert_eq!(st, S1State::default());
    }

    #[test]
    fn test_failed_attach_drops_the_staged_hwpt_and_keeps_an_attached_stage1() {
        let (ok, st, log) = run(on_guest_s1(), true, false, true);
        assert!(!ok);
        assert_eq!(log, ["alloc", "attach 50", "destroy 50"]);
        assert_eq!(st, on_guest_s1());
    }

    #[test]
    fn test_failed_attach_without_a_stage1_parks_on_abort() {
        let (ok, st, log) = run(S1State::default(), true, false, true);
        assert!(!ok);
        assert_eq!(log, ["alloc", "attach 50", "destroy 50", "attach 77"]);
        assert_eq!(st.ste, None);
        assert!(!st.eats);
    }

    #[test]
    fn test_install_failure_action() {
        // Collapsing the two rows into "always park" is the regression
        // this catches: parking a stream whose guest stage-1 is still
        // attached aborts its in-flight DMA.
        assert_eq!(
            install_failure_action(false),
            InstallFailureAction::ParkOnAbort
        );
        assert_eq!(
            install_failure_action(true),
            InstallFailureAction::KeepInstalled
        );
    }

    #[test]
    fn test_resync_command_encodings() {
        // Re-derived from arm-smmu-v3.h: CMDQ_OP_TLBI_NH_ALL = 0x10,
        // CMDQ_OP_ATC_INV = 0x40, CMDQ_ATC_0_SID = [63:32],
        // CMDQ_0_SSV = bit 11, CMDQ_ATC_1_SIZE = [5:0],
        // ATC_INV_SIZE_ALL = 52.
        assert_eq!(CMD_TLBI_NH_ALL, [0x10, 0]);
        let cmd = cmd_atc_inv_all(0x0123);
        assert_eq!(cmd[0] & 0xff, 0x40);
        assert_eq!(cmd[0] >> 32, 0x0123);
        assert_eq!(cmd[0] & (1 << 11), 0, "SSV clear: every substream");
        assert_eq!(cmd[1] & 0x3f, 52);
    }

    #[test]
    fn test_ste_eats_bits() {
        // STE.EATS is word 1 bits [29:28]; bits 27 and 30 must not count,
        // and word 0 never carries EATS.
        for eats in 0u64..=3 {
            let ste = [0xdead_beef_0000_0000u64, eats << 28];
            assert_eq!(ste_eats(&ste), eats != 0, "EATS={eats:#b}");
        }
        assert!(!ste_eats(&[0, 1 << 27]));
        assert!(!ste_eats(&[0, 1 << 30]));
        assert!(!ste_eats(&[u64::MAX, 0]));
    }
}
