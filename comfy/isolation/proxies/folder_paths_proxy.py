from __future__ import annotations
import logging
import os

from typing import Any, Dict, Optional

from pyisolate import ProxiedSingleton

from .base import call_singleton_rpc
from .singleton_contract import SingletonProxyContract, install_singleton_module_proxy

_fp_logger = logging.getLogger(__name__)

FOLDER_PATHS_PUBLIC_CALLABLES = (
    "map_legacy",
    "set_output_directory",
    "set_temp_directory",
    "set_input_directory",
    "get_output_directory",
    "get_temp_directory",
    "get_input_directory",
    "get_user_directory",
    "set_user_directory",
    "get_system_user_directory",
    "get_public_user_directory",
    "get_directory_by_type",
    "filter_files_content_types",
    "annotated_filepath",
    "get_annotated_filepath",
    "exists_annotated_filepath",
    "add_model_folder_path",
    "get_folder_paths",
    "recursive_search",
    "filter_files_extensions",
    "get_full_path",
    "get_full_path_or_raise",
    "get_filename_list_",
    "cached_filename_list_",
    "get_filename_list",
    "get_save_image_path",
    "get_input_subfolders",
)

FOLDER_PATHS_CUSTOM_SYMBOLS = (
    "get_output_directory",
    "get_temp_directory",
    "get_input_directory",
    "get_user_directory",
    "get_annotated_filepath",
    "exists_annotated_filepath",
    "add_model_folder_path",
    "get_folder_paths",
    "get_full_path",
    "get_filename_list",
)

FOLDER_PATHS_RELAY_SYMBOLS = tuple(
    name for name in FOLDER_PATHS_PUBLIC_CALLABLES
    if name not in FOLDER_PATHS_CUSTOM_SYMBOLS
)

FOLDER_PATHS_SINGLETON_CONTRACT = SingletonProxyContract(
    proxy_name="FolderPathsProxy",
    target_name="folder_paths",
    target_public_symbols=FOLDER_PATHS_PUBLIC_CALLABLES,
    relay_symbols=FOLDER_PATHS_RELAY_SYMBOLS,
    custom_symbols=FOLDER_PATHS_CUSTOM_SYMBOLS,
)


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

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        cls._rpc = rpc.create_caller(cls, cls.get_remote_id())

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    @classmethod
    def _get_caller(cls) -> Any:
        if cls._rpc is None:
            raise RuntimeError("FolderPathsProxy RPC caller is not configured")
        return cls._rpc

    def __getattr__(self, name):
        if _is_child_process():
            property_rpc = {
                "models_dir": "rpc_get_models_dir",
                "folder_names_and_paths": "rpc_get_folder_names_and_paths",
                "extension_mimetypes_cache": "rpc_get_extension_mimetypes_cache",
                "filename_list_cache": "rpc_get_filename_list_cache",
            }
            rpc_name = property_rpc.get(name)
            if rpc_name is not None:
                return call_singleton_rpc(self._get_caller(), rpc_name)
            raise AttributeError(name)
        return getattr(_folder_paths(), name)

    def _relay_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        return call_singleton_rpc(self._get_caller(), "rpc_call", method_name, args, kwargs)

    def install_into(self, target_module: Any) -> dict[str, tuple[str, ...]]:
        return install_singleton_module_proxy(
            target_module,
            self,
            FOLDER_PATHS_SINGLETON_CONTRACT,
        )

    @property
    def folder_names_and_paths(self) -> Dict:
        if _is_child_process():
            payload = call_singleton_rpc(self._get_caller(), "rpc_get_folder_names_and_paths")
            return _deserialize_folder_names_and_paths(payload)
        return _folder_paths().folder_names_and_paths

    @property
    def extension_mimetypes_cache(self) -> Dict:
        if _is_child_process():
            return dict(call_singleton_rpc(self._get_caller(), "rpc_get_extension_mimetypes_cache"))
        return dict(_folder_paths().extension_mimetypes_cache)

    @property
    def filename_list_cache(self) -> Dict:
        if _is_child_process():
            return dict(call_singleton_rpc(self._get_caller(), "rpc_get_filename_list_cache"))
        return dict(_folder_paths().filename_list_cache)

    @property
    def models_dir(self) -> str:
        if _is_child_process():
            return str(call_singleton_rpc(self._get_caller(), "rpc_get_models_dir"))
        return _folder_paths().models_dir

    def get_temp_directory(self) -> str:
        if _is_child_process():
            if os.environ.get("PYISOLATE_SANDBOX_MODE") == "required":
                import tempfile

                child_temp = os.path.join(tempfile.gettempdir(), "comfyui_temp")
                os.makedirs(child_temp, exist_ok=True)
                return child_temp
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

    async def rpc_get_models_dir(self) -> str:
        return _folder_paths().models_dir

    async def rpc_get_folder_names_and_paths(self) -> dict[str, dict[str, list[str]]]:
        return _serialize_folder_names_and_paths(_folder_paths().folder_names_and_paths)

    async def rpc_get_extension_mimetypes_cache(self) -> dict[str, Any]:
        return dict(_folder_paths().extension_mimetypes_cache)

    async def rpc_get_filename_list_cache(self) -> dict[str, Any]:
        return dict(_folder_paths().filename_list_cache)

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

    async def rpc_call(self, method_name: str, args: Any, kwargs: Any) -> Any:
        method = getattr(_folder_paths(), method_name)
        return method(*args, **kwargs)
