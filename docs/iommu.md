# Virtual IOMMU

## Rationales

Having the possibility to expose a virtual IOMMU to the guest can be
interesting to support specific use cases. That being said, it is always
important to keep in mind a virtual IOMMU can impact the performance of the
attached devices, which is the reason why one should be careful when enabling
this feature.

### Protect nested virtual machines

The first reason why one might want to expose a virtual IOMMU to the guest is
to increase the security regarding the memory accesses performed by the virtual
devices (VIRTIO devices), on behalf of the guest drivers.

With a virtual IOMMU, the VMM stands between the guest driver and its device
counterpart, validating and translating every address before trying accessing
the guest memory. This is standard interposition that is performed here by the
VMM.

The increased security does not apply for a simple case where we have one VM
per VMM. Because the guest cannot be trusted, as we always consider it could
be malicious and gain unauthorized privileges inside the VM, preventing some
devices from accessing the entire guest memory is pointless.

But let's take the interesting case of nested virtualization, and let's assume
we have a VMM running a first layer VM. This L1 guest is fully trusted as the
user intends to run multiple VMs from this L1. We can end up with multiple L2
VMs running on a single L1 VM. In this particular case, and without exposing a
virtual IOMMU to the L1 guest, it would be possible for any L2 guest to use the
device implementation from the host VMM to access the entire guest L1 memory.
The virtual IOMMU prevents from this kind of trouble as it will validate the
addresses the device is authorized to access.

### Achieve VFIO nested

Another reason for having a virtual IOMMU is to allow passing physical devices
from the host through multiple layers of virtualization. Let's take as example
a system with a physical IOMMU running a VM with a virtual IOMMU. The
implementation of the virtual IOMMU is responsible for updating the physical
DMA Remapping table (DMAR) every time the DMA mapping changes. This must happen
through the VFIO framework on the host as this is the only userspace interface
to interact with a physical IOMMU.

Relying on this update mechanism, it is possible to attach physical devices to
the virtual IOMMU, which allows these devices to be passed from L1 to another
layer of virtualization.

## Why virtio-iommu?

The Cloud Hypervisor project decided to implement the brand new virtio-iommu
device in order to provide a virtual IOMMU to its users. The reason being the
simplicity brought by the paravirtualization solution. By having one side
handled from the guest itself, it removes the complexity of trapping memory
page accesses and shadowing them. This is why the project will not try to
implement a full emulation of a physical IOMMU.

## Pre-requisites

### Kernel

As of Kernel 5.14, virtio-iommu is available for both X86-64 and Aarch64.

## Usage

In order to expose a virtual IOMMU to the guest, it is required to create a
virtio-iommu device and expose it through the ACPI IORT table. This can be
simply achieved by attaching at least one device to the virtual IOMMU.

The way to expose to the guest a specific device as sitting behind this IOMMU
is to explicitly tag it from the command line with the option `iommu=virtio`.

Not all devices support this extra option, and the default value will always
be `off` since we want to avoid the performance impact for most users who don't
need this.

`iommu=on` is accepted as a synonym of `iommu=virtio`, but it is deprecated and
will be removed in a future release.

Refer to the command line `--help` to find out which devices can be supported
to be attached to the virtual IOMMU.

Below is a simple example exposing the `virtio-blk` device as attached to the
virtual IOMMU:

```bash
./cloud-hypervisor \
    --cpus boot=1 \
    --memory size=512M \
    --disk path=focal-server-cloudimg-amd64.raw,iommu=virtio,image_type=raw \
    --kernel custom-vmlinux \
    --cmdline "console=ttyS0 console=hvc0 root=/dev/vda1 rw" \
```

From a guest perspective, it is easy to verify if the device is protected by
the virtual IOMMU. Check the directories listed under
`/sys/kernel/iommu_groups`:

```bash
ls /sys/kernel/iommu_groups
0
```

In this case, only one IOMMU group should be created. Under this group, it is
possible to find out the b/d/f of the device(s) part of this group.

```bash
ls /sys/kernel/iommu_groups/0/devices/
0000:00:03.0
```

And you can validate the device is the one we expect running `lspci`:

```bash
lspci
00:00.0 Host bridge: Intel Corporation Device 0d57
00:01.0 Unassigned class [ffff]: Red Hat, Inc. Device 1057
00:02.0 Unassigned class [ffff]: Red Hat, Inc. Virtio console
00:03.0 Mass storage controller: Red Hat, Inc. Virtio block device
00:04.0 Unassigned class [ffff]: Red Hat, Inc. Virtio RNG
```

### Work with FDT on AArch64

On AArch64 architecture, the virtual IOMMU can still be used even if ACPI is not
enabled. But the effect is different with what the aforementioned test showed.

When ACPI is disabled, virtual IOMMU is supported through Flattened Device Tree
(FDT). In this case, the guest kernel cannot tell which device should be
IOMMU-attached and which should not. No matter how many devices you attached to
the virtual IOMMU by setting `iommu=virtio` option, all the devices on the PCI bus
will be attached to the virtual IOMMU (except the IOMMU itself). Each of the
devices will be added into an IOMMU group.

As a result, the directory content of `/sys/kernel/iommu_groups` would be:

```bash
ls /sys/kernel/iommu_groups/0/devices/
0000:00:02.0
ls /sys/kernel/iommu_groups/1/devices/
0000:00:03.0
ls /sys/kernel/iommu_groups/2/devices/
0000:00:04.0
```

## Faster mappings

By default, the guest memory is mapped with 4k pages and no huge pages, which
causes the virtual IOMMU device to be asked for 4k mappings only. This
configuration slows down the setup of the physical IOMMU as an important number
of requests need to be issued in order to create large mappings.

One use case is even more impacted by the slowdown, the nested VFIO case. When
passing a device through a L2 guest, the VFIO driver running in L1 will update
the DMAR entries for the specific device. Because VFIO pins the entire guest
memory, this means the entire mapping of the L2 guest needs to be stored into
multiple 4k mappings. Obviously, the bigger the L2 guest RAM is, the longer the
update of the mappings will last. There is an additional problem happening in
this case, if the L2 guest RAM is quite large, it will require a large number
of mappings, which might exceed the VFIO limit set on the host. The default
value is 65536, which can simply be reached with a 256MiB sized RAM.

The way to solve both problems, the slowdown and the limit being exceeded, is
to reduce the amount of requests to describe those same large mappings. This
can be achieved by using 2MiB pages, known as huge pages. By seeing the guest
RAM as larger pages, and because the virtual IOMMU device supports it, the
guest will require less mappings, which will prevent the limit from being
exceeded, but also will take less time to process them on the host. That's
how using huge pages as much as possible can speed up VM boot time.

### Basic usage

Let's look at an example of how to run a guest with huge pages.

First, make sure your system has enough pages to cover the entire guest RAM:
```bash
# This example creates 4096 hugepages
echo 4096 > /proc/sys/vm/nr_hugepages
```

Next step is simply to create the VM. Two things are important, first we want
the VM RAM to be mapped on huge pages by backing it with `/dev/hugepages`. And
second thing, we need to create some huge pages in the guest itself so they can
be consumed.

```bash
./cloud-hypervisor \
    --cpus boot=1 \
    --memory size=8G,hugepages=on \
    --disk path=focal-server-cloudimg-amd64.raw,image_type=raw \
    --kernel custom-vmlinux \
    --cmdline "console=ttyS0 console=hvc0 root=/dev/vda1 rw hugepagesz=2M hugepages=2048" \
    --net tap=,mac=,iommu=virtio
```

### Nested usage

Let's now look at the specific example of nested virtualization. In order to
reach optimized performances, the L2 guest also needs to be mapped based on
huge pages. Here is how to achieve this, assuming the physical device you are
passing through is `0000:00:01.0`.

```bash
./cloud-hypervisor \
    --cpus boot=1 \
    --memory size=8G,hugepages=on \
    --disk path=focal-server-cloudimg-amd64.raw,image_type=raw \
    --kernel custom-vmlinux \
    --cmdline "console=ttyS0 console=hvc0 root=/dev/vda1 rw kvm-intel.nested=1 vfio_iommu_type1.allow_unsafe_interrupts rw hugepagesz=2M hugepages=2048" \
    --device path=/sys/bus/pci/devices/0000:00:01.0,iommu=virtio
```

Once the L1 VM is running, unbind the device from the default driver in the
guest, and bind it to VFIO (it should appear as `0000:00:04.0`).

```bash
echo 0000:00:04.0 > /sys/bus/pci/devices/0000\:00\:04.0/driver/unbind
echo 8086 1502 > /sys/bus/pci/drivers/vfio-pci/new_id
echo 0000:00:04.0 > /sys/bus/pci/drivers/vfio-pci/bind
```

Last thing is to start the L2 guest with the huge pages memory backend.

```bash
./cloud-hypervisor \
    --cpus boot=1 \
    --memory size=4G,hugepages=on \
    --disk path=focal-server-cloudimg-amd64.raw,image_type=raw \
    --kernel custom-vmlinux \
    --cmdline "console=ttyS0 console=hvc0 root=/dev/vda1 rw" \
    --device path=/sys/bus/pci/devices/0000:00:04.0
```

### Dedicated IOMMU PCI segments

To facilitate hotplug of devices that require being behind an IOMMU it is
possible to mark entire PCI segments as behind the IOMMU.

This is accomplished through `--platform
num_pci_segments=<number_of_segments>,iommu_segments=<range of segments>` or
via the equivalents in `PlatformConfig` for the API.

e.g.

```bash
./cloud-hypervisor \
    --api-socket=/tmp/api \
    --cpus boot=1 \
    --memory size=4G,hugepages=on \
    --disk path=focal-server-cloudimg-amd64.raw,image_type=raw \
    --kernel custom-vmlinux \
    --cmdline "console=ttyS0 console=hvc0 root=/dev/vda1 rw" \
    --platform num_pci_segments=2,iommu_segments=1
```

This adds a second PCI segment to the platform behind the IOMMU. A VFIO device
requiring the IOMMU then may be hotplugged:

e.g.

```bash
./ch-remote --api-socket=/tmp/api add-device path=/sys/bus/pci/devices/0000:00:04.0,iommu=virtio,pci_segment=1
```

Devices that cannot be placed behind an IOMMU (e.g. lacking an `iommu=` option)
cannot be placed on the IOMMU segments.



## Emulated ARM SMMUv3 (AArch64)

On AArch64, a passthrough device placed behind `iommu=smmuv3` sits behind an
emulated ARM SMMUv3 instead of the virtio-iommu. Unlike virtio-iommu, it is a
full device model of real hardware, so a guest uses its stock `arm-smmu-v3`
driver with no paravirtualized interface.

Its purpose is nested translation. The guest programs stage-1 translation in the
emulated SMMUv3, and the VMM offloads it to the physical SMMUv3 through iommufd
nested page tables, rather than shadowing the guest's page tables. This makes it
possible to assign devices that drive the IOMMU themselves, such as NVIDIA
Grace-Blackwell GPUs, which need PASID and ATS to reach the physical SMMUv3.

One emulated SMMUv3 is created per physical SMMUv3 backing an assigned device,
so a guest given devices behind different physical SMMUv3s sees one instance
per host IOMMU.

Requirements:

- AArch64, KVM, and a host kernel with iommufd nested translation support for
  ARM SMMUv3.
- `--platform iommufd=on`, since the offload goes through iommufd. Requesting
  `iommu=smmuv3` without it is rejected.
- Only passthrough devices can be placed behind it. Every other device keeps
  using `iommu=virtio`.

Limitations:

- Only devices assigned with `--device` can sit behind it. Asking for
  `iommu=smmuv3` on any other device is rejected.
- A guest is given a single type of virtual IOMMU. Mixing `iommu=smmuv3` and
  `iommu=virtio` across devices is rejected, and so is combining it with the
  `iommu_segments` option of `--platform`, which implies a virtio-iommu.
- Devices cannot be hotplugged behind it, they have to be assigned at boot.
- The guest must boot through UEFI firmware. Only the ACPI IORT places a
  device behind the emulated SMMUv3; a direct kernel boot describes the
  platform by device tree, so it is refused when a device asks for
  `iommu=smmuv3`.
- A device must be on bus 0 of its PCI segment, as every Cloud Hypervisor
  device is: its StreamID is `256 * segment + devfn`, the same value as its
  ITS DeviceID.
- Snapshot, restore and live migration are not supported. A snapshot does
  capture the emulated SMMUv3 registers, but they are not applied back on
  restore, and the nested translation set up through iommufd is not rebuilt.

```bash
./cloud-hypervisor \
    --api-socket=/tmp/api \
    --cpus boot=8 \
    --memory size=16G \
    --disk path=focal-server-cloudimg-arm64.raw \
    --kernel CLOUDHV_EFI.fd \
    --platform iommufd=on \
    --device path=/sys/bus/pci/devices/0009:01:00.0/,iommu=smmuv3
```

### Capabilities and placement

The emulated SMMUv3 advertises what its host SMMUv3 backs: coherency, ATS,
the substream ID width, the stage-1 table formats and granules, the output
address size (at most 48 bits), range invalidation, the break-before-make
level and 52-bit virtual addresses. A guest kernel checks the last two before
it shares its page tables with the SMMU (SVA), which CUDA on a Grace GPU
requires.

A device is offered a PASID capability when the host reports a PASID width
for it (`IOMMU_GET_HW_INFO`, Linux 6.15 and later). On an older host kernel,
which reports none, the capability is taken from the device's physical one
when the host has enabled it, as the host IOMMU driver does only when it can
back PASID. A device with a PASID capability is presented as a root complex
integrated endpoint.

The emulated SMMUv3s sit in a 16 MiB MMIO window at `0x0E00_0000`, one
128 KiB frame each, so a guest can have one per host SMMU of a large host.

### Tegra241 CMDQV

Built with the `smmuv3-accel` feature (reported by `vmm.ping`), an emulated
SMMUv3 whose host SMMU carries a compatible NVIDIA Tegra241 CMDQV is given one
too, described in the DSDT as `NVDA200C`. The guest then issues invalidations
through hardware queues instead of trapping each command into the VMM. Each
CMDQV sits in a 64 MiB window at `0x0A00_0000`, and the host's VINTF page is
mapped into the guest so queue doorbells need no exit.

A hardware queue's memory must be physically contiguous, so the command queue
size the SMMUv3 advertises is capped by how guest memory is backed: the host
page size, or the hugepage size when every memory zone uses hugepages of an
explicit size.

### Grace GPUs

The NVIDIA driver for a Grace GPU with coherent memory (GH200, GB200, GB300)
requires the GPU at guest device and function `00.0`, and 8 memory-less guest
NUMA nodes with an SRAT Generic Initiator naming it, into which it onlines the
GPU memory. Device 0 of PCI segment 0 is the host bridge, so put each GPU on
its own segment with `pci_device_id=0`, and name it in `--numa` with
`device_id`:

```bash
--platform iommufd=on,num_pci_segments=2 \
--device path=/sys/bus/pci/devices/0009:01:00.0/,iommu=smmuv3,pci_segment=1,pci_device_id=0,id=gpu0 \
--memory size=0 \
--memory-zone id=mem0,size=64G \
--numa guest_numa_id=0,cpus=[0-15],memory_zones=[mem0] \
       guest_numa_id=1,device_id=gpu0 guest_numa_id=2,device_id=gpu0 \
       guest_numa_id=3,device_id=gpu0 guest_numa_id=4,device_id=gpu0 \
       guest_numa_id=5,device_id=gpu0 guest_numa_id=6,device_id=gpu0 \
       guest_numa_id=7,device_id=gpu0 guest_numa_id=8,device_id=gpu0
```
