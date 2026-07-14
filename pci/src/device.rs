// Copyright 2018 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::any::Any;
use std::sync::{Arc, Barrier};
use std::{io, result};

use log::warn;
use thiserror::Error;
use vm_allocator::{AddressAllocator, SystemAllocator};
use vm_device::{BusDeviceSync, Resource};

use crate::PciBarConfiguration;
use crate::configuration::{self, PciBarRegionType};

#[derive(Error, Debug)]
pub enum Error {
    /// Setup of the device capabilities failed.
    #[error("Setup of the device capabilities failed")]
    CapabilitiesSetup(#[source] configuration::Error),
    /// Allocating space for an IO BAR failed.
    #[error("Allocating space for an IO BAR of size {0} failed")]
    IoAllocationFailed(u64),
    /// Registering an IO BAR failed.
    #[error("Registering an IO BAR at address {0:#x} failed")]
    IoRegistrationFailed(u64, #[source] configuration::Error),
    /// Expected resource not found.
    #[error("Expected resource not found")]
    MissingResource,
    /// Invalid resource.
    #[error("Invalid resource: {0:?}")]
    InvalidResource(Resource),
}
pub type Result<T> = result::Result<T, Error>;

/// Describes a BAR mapping that must be released (its old guest-physical
/// location torn down) because the guest reprogrammed the BAR address while
/// the decode bit of its space was off, or as the release half of an
/// immediate (decode-on) relocation.
#[derive(Clone, Copy, Debug)]
pub struct ReleaseParams {
    /// The BAR slot being released (the low/primary slot for a 64-bit BAR,
    /// ROM_BAR_IDX for the expansion ROM).
    pub bar_idx: usize,
    /// The address the BAR is currently materialized at.
    pub base: u64,
    pub len: u64,
    pub region_type: PciBarRegionType,
}

/// Describes a BAR mapping that must be installed at its new guest-physical
/// location. The device-side commit derives the released-from
/// address from its own records, which the release is forbidden to mutate;
/// `old_base` travels here only to anchor the *allocator* to the BAR's
/// PCI segment (see the field doc) and is not consumed by the device side.
#[derive(Clone, Copy, Debug)]
pub struct InstallParams {
    /// The BAR slot being installed (the low/primary slot for a 64-bit BAR,
    /// ROM_BAR_IDX for the expansion ROM).
    pub bar_idx: usize,
    /// The address the BAR was released from. Its MMIO allocator window is
    /// the device's PCI segment aperture: a device cannot decode outside its
    /// own segment, so the install must allocate `new_base` from that same
    /// window and reject a target that falls outside it.
    pub old_base: u64,
    pub new_base: u64,
    pub len: u64,
    pub region_type: PciBarRegionType,
}

/// A relocation plan emitted by `write_config_register`. Because release and
/// install happen on *different* config-register writes when the decode bit
/// is off (release on the BAR address change, install on the decode-enable
/// edge), a single write can produce releases, installs, or both (the
/// immediate decode-on case).
#[derive(Clone, Debug, Default)]
pub struct BarRelocation {
    pub release: Vec<ReleaseParams>,
    pub install: Vec<InstallParams>,
}

impl BarRelocation {
    pub fn is_empty(&self) -> bool {
        self.release.is_empty() && self.install.is_empty()
    }
}

/// Outcome of an attempted BAR install, reported back to the device so its
/// config-space bookkeeping can follow what actually happened.
#[derive(Clone, Copy, Debug)]
pub enum BarRelocationStatus {
    /// The BAR is now mapped at `base`.
    Applied { base: u64 },
    /// The install failed and the BAR is currently not mapped anywhere. It
    /// stays pending and is retried at the guest's next decode-enable edge.
    Pending,
}

pub trait PciDevice: Send {
    /// Allocates the needed PCI BARs space using the `allocate` function which takes a size and
    /// returns an address. Returns a Vec of (GuestAddress, GuestUsize) tuples.
    fn allocate_bars(
        &mut self,
        _allocator: &mut SystemAllocator,
        _mmio32_allocator: &mut AddressAllocator,
        _mmio64_allocator: &mut AddressAllocator,
        _resources: Option<Vec<Resource>>,
    ) -> Result<Vec<PciBarConfiguration>> {
        Ok(Vec::new())
    }

    /// Frees the PCI BARs previously allocated with a call to allocate_bars().
    fn free_bars(
        &mut self,
        _allocator: &mut SystemAllocator,
        _mmio32_allocator: &mut AddressAllocator,
        _mmio64_allocator: &mut AddressAllocator,
    ) -> Result<()> {
        Ok(())
    }

    /// Sets a register in the configuration space.
    /// * `reg_idx` - The index of the config register to modify.
    /// * `offset` - Offset into the register.
    fn write_config_register(
        &mut self,
        reg_idx: usize,
        offset: u64,
        data: &[u8],
    ) -> (BarRelocation, Option<Arc<Barrier>>);
    /// Gets a register from the configuration space.
    /// * `reg_idx` - The index of the config register to read.
    fn read_config_register(&mut self, reg_idx: usize) -> u32;
    /// Reads from a BAR region mapped into the device.
    /// * `addr` - The guest address inside the BAR.
    /// * `data` - Filled with the data from `addr`.
    fn read_bar(&mut self, _base: u64, _offset: u64, _data: &mut [u8]) {}
    /// Writes to a BAR region mapped into the device.
    /// * `addr` - The guest address inside the BAR.
    /// * `data` - The data to write.
    fn write_bar(&mut self, _base: u64, _offset: u64, _data: &[u8]) -> Option<Arc<Barrier>> {
        None
    }
    /// Relocation release side: tear down the device's host-side state
    /// backing the BAR (KVM memslots, VFIO DMA maps) at its OLD address.
    /// Must NOT mutate the recorded base -- it stays keyed by index so
    /// `move_bar_commit` can still derive per-region offsets from it.
    ///
    /// The default no-op suits pure trap-emulated BARs; a device with KVM
    /// memslots, VFIO DMA or userspace mappings MUST implement this, or it
    /// silently reintroduces the overlap this split prevents.
    fn move_bar_prepare(&mut self, _bar_idx: usize) -> result::Result<(), io::Error> {
        Ok(())
    }
    /// Relocation install side: set up the device's host-side state at
    /// `new_base` and update the recorded base. The BAR is identified by
    /// slot index, not its current address (which mutates under the guest);
    /// implementations derive the old address from their own records.
    fn move_bar_commit(
        &mut self,
        _bar_idx: usize,
        _new_base: u64,
    ) -> result::Result<(), io::Error> {
        Ok(())
    }
    /// Reports the outcome of an attempted BAR install so the device can
    /// reconcile its config-space bookkeeping with what actually happened.
    fn on_bar_relocation_status(&mut self, bar_idx: usize, status: BarRelocationStatus) {
        warn!("BAR {bar_idx} relocation status dropped: {status:?}");
    }
    /// Provides a mutable reference to the Any trait. This is useful to let
    /// the caller have access to the underlying type behind the trait.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Optionally returns a unique identifier.
    fn id(&self) -> Option<String>;
}

/// This trait defines a set of functions which can be triggered whenever a
/// PCI device is modified in any way.
pub trait DeviceRelocation: Send + Sync {
    /// Release the OLD guest-physical mapping of a BAR: frees the allocator
    /// range, removes the trap-emulated bus range, tears down the virtio
    /// shm / ioeventfd old-side mapping and runs the device-side
    /// `move_bar_prepare`. The bus handle stays stored in the `PciBus`
    /// device pair; the matching install re-inserts it.
    fn move_bar_prepare(
        &self,
        params: &ReleaseParams,
        pci_dev: &mut dyn PciDevice,
    ) -> result::Result<(), io::Error>;

    /// Install the NEW guest-physical mapping of a BAR previously released
    /// by `move_bar_prepare`: allocator range, bus insertion (using
    /// `bus_device`, the MMIO/IO bus handle the PCI bus stores alongside the
    /// device), the device-side commit and the follow-on mappings.
    fn move_bar_commit(
        &self,
        params: &InstallParams,
        pci_dev: &mut dyn PciDevice,
        bus_device: &Arc<dyn BusDeviceSync>,
    ) -> result::Result<(), io::Error>;
}
