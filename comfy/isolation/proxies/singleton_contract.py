from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SingletonProxyContract:
    proxy_name: str
    target_name: str
    target_public_symbols: tuple[str, ...]
    relay_symbols: tuple[str, ...] = ()
    custom_symbols: tuple[str, ...] = ()
    unsupported_symbols: tuple[str, ...] = ()

    @property
    def covered_symbols(self) -> frozenset[str]:
        return frozenset(
            self.relay_symbols + self.custom_symbols + self.unsupported_symbols
        )

    def validate(self) -> None:
        duplicate_symbols = _duplicates(
            self.relay_symbols + self.custom_symbols + self.unsupported_symbols
        )
        if duplicate_symbols:
            raise SingletonProxyContractError(
                f"{self.proxy_name} duplicate singleton contract symbols: "
                f"{', '.join(duplicate_symbols)}"
            )

        target_symbols = frozenset(self.target_public_symbols)
        missing = sorted(target_symbols - self.covered_symbols)
        stale = sorted(self.covered_symbols - target_symbols)
        if missing or stale:
            raise SingletonProxyContractError(
                format_contract_drift(
                    proxy_name=self.proxy_name,
                    target_name=self.target_name,
                    missing_symbols=missing,
                    stale_symbols=stale,
                )
            )


class SingletonProxyContractError(RuntimeError):
    pass


def format_contract_drift(
    *,
    proxy_name: str,
    target_name: str,
    missing_symbols: Iterable[str],
    stale_symbols: Iterable[str],
) -> str:
    missing = tuple(missing_symbols)
    stale = tuple(stale_symbols)
    parts = [
        f"{proxy_name} singleton proxy contract drift for {target_name}.",
        "Required action for each symbol: relay, custom serialization, or unsupported classification.",
    ]
    if missing:
        parts.append("Missing target symbols: " + ", ".join(missing))
    if stale:
        parts.append("Stale contract symbols: " + ", ".join(stale))
    return " ".join(parts)


def fail_unsupported_singleton_symbol(proxy_name: str, symbol_name: str) -> None:
    raise RuntimeError(
        f"{proxy_name}.{symbol_name} is intentionally unsupported by the "
        "singleton proxy contract. Required action: relay, custom serialization, "
        "or unsupported classification."
    )


def install_singleton_module_proxy(
    target_module: Any,
    proxy: Any,
    contract: SingletonProxyContract,
) -> dict[str, tuple[str, ...]]:
    contract.validate()

    for name in contract.relay_symbols:
        setattr(target_module, name, _make_relay_wrapper(proxy, contract.proxy_name, name))

    for name in contract.custom_symbols:
        setattr(target_module, name, _get_explicit_proxy_member(proxy, name, contract.proxy_name))

    for name in contract.unsupported_symbols:
        setattr(target_module, name, make_unsupported_singleton_wrapper(contract.proxy_name, name))

    return {
        "relay": contract.relay_symbols,
        "custom": contract.custom_symbols,
        "unsupported": contract.unsupported_symbols,
    }


def make_unsupported_singleton_wrapper(proxy_name: str, symbol_name: str) -> Callable[..., Any]:
    def unsupported_singleton_wrapper(*args: Any, **kwargs: Any) -> Any:
        fail_unsupported_singleton_symbol(proxy_name, symbol_name)

    unsupported_singleton_wrapper.__name__ = symbol_name
    unsupported_singleton_wrapper.__qualname__ = f"{proxy_name}.{symbol_name}"
    return unsupported_singleton_wrapper


def _make_relay_wrapper(proxy: Any, proxy_name: str, symbol_name: str) -> Callable[..., Any]:
    relay_call = getattr(proxy, "_relay_call", None)
    if relay_call is None:
        raise SingletonProxyContractError(
            f"{proxy_name} cannot relay {symbol_name}: _relay_call is missing"
        )

    def singleton_relay_wrapper(*args: Any, **kwargs: Any) -> Any:
        return relay_call(symbol_name, *args, **kwargs)

    singleton_relay_wrapper.__name__ = symbol_name
    singleton_relay_wrapper.__qualname__ = f"{proxy_name}.{symbol_name}"
    return singleton_relay_wrapper


def _get_explicit_proxy_member(proxy: Any, symbol_name: str, proxy_name: str) -> Any:
    for cls in type(proxy).mro():
        if symbol_name in cls.__dict__:
            return getattr(proxy, symbol_name)
    raise SingletonProxyContractError(
        f"{proxy_name}.{symbol_name} is marked custom but has no explicit proxy member"
    )


def _duplicates(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for symbol in symbols:
        if symbol in seen:
            duplicates.append(symbol)
        seen.add(symbol)
    return sorted(set(duplicates))
