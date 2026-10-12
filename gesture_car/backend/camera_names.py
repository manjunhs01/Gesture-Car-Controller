"""Friendly webcam names on Windows (for ``--list-cameras``), without extra packages.

Lists the DirectShow video-input devices in the order OpenCV's DirectShow backend numbers them, so ``names[i]`` is the name
of ``--camera i`` when that camera is opened through DirectShow. Uses plain ``ctypes`` COM calls. Any failure (another OS,
COM error, no devices) returns an empty list: names are a nice extra, never a requirement.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import uuid
from typing import Any

log = logging.getLogger(__name__)

_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_CLSCTX_INPROC_SERVER = 1
_COINIT_APARTMENTTHREADED = 2
_VT_BSTR = 8
_S_OK = 0

# vtable slots (IUnknown = 0..2)
_RELEASE = 2
_CREATE_CLASS_ENUMERATOR = 3  # ICreateDevEnum
_ENUM_NEXT = 3  # IEnumMoniker
_BIND_TO_STORAGE = 9  # IMoniker (IUnknown 3 + IPersist 1 + IPersistStream 4 + BindToObject)
_BAG_READ = 3  # IPropertyBag


def _guid(text: str) -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(uuid.UUID(text).bytes_le, 16)


def _method(obj: ctypes.c_void_p, slot: int, *argtypes: Any) -> Any:
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
    return prototype(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, _RELEASE)(obj)


def _enumerate() -> list[str]:
    ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
    oleaut32 = ctypes.windll.oleaut32  # type: ignore[attr-defined]
    initialised = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED) >= 0
    names: list[str] = []
    dev_enum = ctypes.c_void_p()
    enum_moniker = ctypes.c_void_p()
    try:
        hr = ole32.CoCreateInstance(
            _guid(_CLSID_SYSTEM_DEVICE_ENUM), None, _CLSCTX_INPROC_SERVER, _guid(_IID_ICREATE_DEV_ENUM), ctypes.byref(dev_enum)
        )
        if hr != _S_OK:
            return names
        create = _method(dev_enum, _CREATE_CLASS_ENUMERATOR, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if create(dev_enum, _guid(_CLSID_VIDEO_INPUT_CATEGORY), ctypes.byref(enum_moniker), 0) != _S_OK or not enum_moniker:
            return names  # S_FALSE: no video devices
        nxt = _method(enum_moniker, _ENUM_NEXT, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)
        while True:
            moniker = ctypes.c_void_p()
            if nxt(enum_moniker, 1, ctypes.byref(moniker), None) != _S_OK or not moniker:
                break
            bag = ctypes.c_void_p()
            try:
                bind = _method(moniker, _BIND_TO_STORAGE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
                name = ""
                if bind(moniker, None, None, _guid(_IID_IPROPERTY_BAG), ctypes.byref(bag)) == _S_OK and bag:
                    variant = ctypes.create_string_buffer(24)  # VARIANT: vt at 0, BSTR pointer at 8 (64-bit) / 8 (32-bit)
                    read = _method(bag, _BAG_READ, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p)
                    if read(bag, "FriendlyName", variant, None) == _S_OK:
                        vt = ctypes.c_ushort.from_buffer(variant, 0).value
                        ptr = ctypes.c_void_p.from_buffer(variant, 8).value
                        if vt == _VT_BSTR and ptr:
                            name = ctypes.wstring_at(ptr)
                        oleaut32.VariantClear(variant)
                names.append(name)
            finally:
                _release(bag)
                _release(moniker)
    finally:
        _release(enum_moniker)
        _release(dev_enum)
        if initialised:
            ole32.CoUninitialize()
    return names


def directshow_camera_names(timeout_s: float = 3.0) -> list[str]:
    """Names of the DirectShow cameras, index-aligned with ``cv2.CAP_DSHOW``; ``[]`` if unknown."""
    if not sys.platform.startswith("win"):
        return []
    result: list[str] = []

    def work() -> None:  # own thread: its own COM apartment, and a stuck driver cannot block the caller
        try:
            result.extend(_enumerate())
        except Exception:
            log.debug("Could not read the camera names", exc_info=True)

    thread = threading.Thread(target=work, name="camera-names", daemon=True)
    thread.start()
    thread.join(timeout_s)
    return list(result) if not thread.is_alive() else []
