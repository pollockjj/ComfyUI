from __future__ import annotations
import os
from typing import Any, Dict, Optional

from pyisolate import ProxiedSingleton

from .base import call_singleton_rpc


def _folder_paths():
    import folder_paths

    return folder_paths


def _is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def _serialize_folder_names_and_paths(data: dict[str, tuple[list[str], set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        key: {"paths": list(paths), "extensions": sorted(list(extensions))}
        for key, (paths, extensions) in data.items()
    }


def _deserialize_folder_names_and_paths(data: dict[str, dict[str, list[str]]]) -> dict[str, tuple[list[str], set[str]]]:
    return {
        key: (list(value.get("paths", [])), set(value.get("extensions", [])))
        for key, value in data.items()
    }


class FolderPathsProxy(ProxiedSingleton):
    """
    Dynamic proxy for folder_paths.
    Uses __getattr__ for most lookups, with explicit handling for
    mutable collections to ensure efficient by-value transfer.
    """

    _rpc: Optional[Any] = None
    _snapshot_cache: Optional[dict[str, Any]] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        cls._rpc = rpc.create_caller(cls, cls.get_remote_id())
        cls._snapshot_cache = None

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None
        cls._snapshot_cache = None

    @classmethod
    def _get_caller(cls) -> Any:
        if cls._rpc is None:
            raise RuntimeError("FolderPathsProxy RPC caller is not configured")
        return cls._rpc

    @classmethod
    def _get_snapshot(cls, force_refresh: bool = False) -> dict[str, Any]:
        if not _is_child_process():
            fp = _folder_paths()
            return {
                "models_dir": fp.models_dir,
                "input_directory": fp.get_input_directory(),
                "output_directory": fp.get_output_directory(),
                "temp_directory": fp.get_temp_directory(),
                "user_directory": fp.get_user_directory(),
                "supported_pt_extensions": sorted(list(fp.supported_pt_extensions)),
                "folder_names_and_paths": _serialize_folder_names_and_paths(fp.folder_names_and_paths),
                "extension_mimetypes_cache": dict(fp.extension_mimetypes_cache),
                "filename_list_cache": dict(fp.filename_list_cache),
            }
        if force_refresh or cls._snapshot_cache is None:
            cls._snapshot_cache = call_singleton_rpc(cls._get_caller(), "rpc_snapshot")
        return cls._snapshot_cache

    def __getattr__(self, name):
        if _is_child_process():
            snapshot = self._get_snapshot()
            if name in snapshot:
                return snapshot[name]
            raise AttributeError(name)
        return getattr(_folder_paths(), name)

    @property
    def folder_names_and_paths(self) -> Dict:
        snapshot = self._get_snapshot()
        return _deserialize_folder_names_and_paths(snapshot["folder_names_and_paths"])

    @property
    def extension_mimetypes_cache(self) -> Dict:
        snapshot = self._get_snapshot()
        return dict(snapshot["extension_mimetypes_cache"])

    @property
    def filename_list_cache(self) -> Dict:
        snapshot = self._get_snapshot()
        return dict(snapshot["filename_list_cache"])

    @property
    def models_dir(self) -> str:
        if _is_child_process():
            return str(self._get_snapshot()["models_dir"])
        return _folder_paths().models_dir

    def get_temp_directory(self) -> str:
        if _is_child_process():
            return call_singleton_rpc(self._get_caller(), "rpc_get_temp_directory")
        return _folder_paths().get_temp_directory()

    def get_input_directory(self) -> str:
        if _is_child_process():
            return call_singleton_rpc(self._get_caller(), "rpc_get_input_directory")
        return _folder_paths().get_input_directory()

    def get_output_directory(self) -> str:
        if _is_child_process():
            return call_singleton_rpc(self._get_caller(), "rpc_get_output_directory")
        return _folder_paths().get_output_directory()

    def get_user_directory(self) -> str:
        if _is_child_process():
            return call_singleton_rpc(self._get_caller(), "rpc_get_user_directory")
        return _folder_paths().get_user_directory()

    def get_annotated_filepath(self, name: str, default_dir: str | None = None) -> str:
        if _is_child_process():
            return call_singleton_rpc(
                self._get_caller(), "rpc_get_annotated_filepath", name, default_dir
            )
        return _folder_paths().get_annotated_filepath(name, default_dir)

    def exists_annotated_filepath(self, name: str) -> bool:
        if _is_child_process():
            return bool(
                call_singleton_rpc(self._get_caller(), "rpc_exists_annotated_filepath", name)
            )
        return bool(_folder_paths().exists_annotated_filepath(name))

    def add_model_folder_path(
        self, folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        if _is_child_process():
            call_singleton_rpc(
                self._get_caller(),
                "rpc_add_model_folder_path",
                folder_name,
                full_folder_path,
                is_default,
            )
            self.__class__._snapshot_cache = None
            return None
        _folder_paths().add_model_folder_path(folder_name, full_folder_path, is_default)
        return None

    def get_folder_paths(self, folder_name: str) -> list[str]:
        if _is_child_process():
            return list(call_singleton_rpc(self._get_caller(), "rpc_get_folder_paths", folder_name))
        return list(_folder_paths().get_folder_paths(folder_name))

    def get_filename_list(self, folder_name: str) -> list[str]:
        if _is_child_process():
            return list(call_singleton_rpc(self._get_caller(), "rpc_get_filename_list", folder_name))
        return list(_folder_paths().get_filename_list(folder_name))

    def get_full_path(self, folder_name: str, filename: str) -> str | None:
        if _is_child_process():
            return call_singleton_rpc(self._get_caller(), "rpc_get_full_path", folder_name, filename)
        return _folder_paths().get_full_path(folder_name, filename)

    async def rpc_snapshot(self) -> dict[str, Any]:
        return self.__class__._get_snapshot(force_refresh=True)

    async def rpc_get_temp_directory(self) -> str:
        return _folder_paths().get_temp_directory()

    async def rpc_get_input_directory(self) -> str:
        return _folder_paths().get_input_directory()

    async def rpc_get_output_directory(self) -> str:
        return _folder_paths().get_output_directory()

    async def rpc_get_user_directory(self) -> str:
        return _folder_paths().get_user_directory()

    async def rpc_get_annotated_filepath(self, name: str, default_dir: str | None = None) -> str:
        return _folder_paths().get_annotated_filepath(name, default_dir)

    async def rpc_exists_annotated_filepath(self, name: str) -> bool:
        return _folder_paths().exists_annotated_filepath(name)

    async def rpc_add_model_folder_path(
        self, folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        _folder_paths().add_model_folder_path(folder_name, full_folder_path, is_default)

    async def rpc_get_folder_paths(self, folder_name: str) -> list[str]:
        return _folder_paths().get_folder_paths(folder_name)

    async def rpc_get_filename_list(self, folder_name: str) -> list[str]:
        return _folder_paths().get_filename_list(folder_name)

    async def rpc_get_full_path(self, folder_name: str, filename: str) -> str | None:
        return _folder_paths().get_full_path(folder_name, filename)
