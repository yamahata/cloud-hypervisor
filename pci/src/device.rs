// Copyright 2018 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::any::Any;
use std::sync::{Arc, Barrier};
use std::{io, result};

use serde::{Deserialize, Serialize};
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
pub(crate) type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BarReprogrammingParams {
    #[serde(default)]
    pub bar_idx: Option<usize>,
    pub old_base: u64,
    pub new_base: u64,
    pub len: u64,
    pub region_type: PciBarRegionType,
}

/// A BAR to release: its current guest-physical location is torn down.
#[derive(Clone, Copy, Debug)]
pub struct ReleaseParams {
    /// The BAR slot (the low/primary slot for a 64-bit BAR, ROM_BAR_IDX for
    /// the expansion ROM).
    pub bar_idx: usize,
}

/// A BAR to install at `new_base`, the guest's current config-space target.
#[derive(Clone, Copy, Debug)]
pub struct InstallParams {
    /// The BAR slot (the low/primary slot for a 64-bit BAR, ROM_BAR_IDX for
    /// the expansion ROM).
    pub bar_idx: usize,
    pub new_base: u64,
}

// A relocation plan emitted by `write_config_register`.
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
    /// Tear down the device's host-side state backing the BAR
    /// (KVM memslots, VFIO DMA maps) at its old BAR base address.
    /// Must NOT mutate the recorded base so that `move_bar_commit`
    /// can still derive per-region offsets from it.
    fn move_bar_prepare(&mut self, _bar_idx: usize) -> result::Result<(), io::Error> {
        Ok(())
    }
    /// Set up the device's host-side state at `new_base` and update the recorded base.
    /// The implementations derive the old address from their own records.
    fn move_bar_commit(
        &mut self,
        _bar_idx: usize,
        _new_base: u64,
    ) -> result::Result<(), io::Error> {
        Ok(())
    }
    /// BAR `bar_idx` is now mapped at its config-space address. A failed
    /// install reports nothing: the slot stays released and is retried, while
    /// its space is decoded, on this device's next BAR or COMMAND write and
    /// after any other device's BAR release.
    fn on_bar_installed(&mut self, _bar_idx: usize) {}
    /// Provides a mutable reference to the Any trait. This is useful to let
    /// the caller have access to the underlying type behind the trait.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Optionally returns a unique identifier.
    fn id(&self) -> Option<String>;
}

/// This trait defines a set of functions which can be triggered whenever a
/// PCI device is modified in any way.
pub trait DeviceRelocation: Send + Sync {
    /// Release the OLD guest-physical mapping of a BAR being relocated.
    ///
    /// This frees the allocator range, removes the trap-emulated bus range,
    /// tears down the virtio shm / ioeventfd old-side mapping and runs the
    /// device-side `move_bar_prepare`. The bus handle itself stays stored in
    /// the `PciBus` device pair; the matching install re-inserts it at the
    /// new address.
    fn move_bar_prepare(
        &self,
        pci_dev: &mut dyn PciDevice,
        params: &ReleaseParams,
    ) -> result::Result<(), io::Error>;

    /// Install the NEW guest-physical mapping of a BAR previously released
    /// by `move_bar_prepare`: allocator range, bus insertion (using
    /// `bus_device`, the MMIO/IO bus handle the PCI bus stores alongside the
    /// device), the device-side commit and the follow-on mappings.
    fn move_bar_commit(
        &self,
        pci_dev: &mut dyn PciDevice,
        bus_device: &Arc<dyn BusDeviceSync>,
        params: &InstallParams,
    ) -> result::Result<(), io::Error>;

    /// Release every BAR of `pci_dev` whose address space is not decoded
    /// (memory or I/O space disabled in its COMMAND register). Called once
    /// after a restore: such a BAR is installed at its config-space target
    /// once the guest decodes its space, so no in-flight move is replayed.
    /// The default does nothing.
    fn release_undecoded_bars(
        &self,
        _pci_dev: &mut dyn PciDevice,
    ) -> result::Result<(), io::Error> {
        Ok(())
    }
}
