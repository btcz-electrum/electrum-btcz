# Optional development fee for BTCZ Electrum.
# 0.1% of the amount sent, shown as its own line in the Confirm Transaction dialog,
# and the user can switch it off there (remembered in config key 'btcz_dev_fee').
# No fee on sends below DEV_FEE_MIN_SEND or on sends to the wallet's own addresses.
from typing import List, Sequence, Optional

from .bitcoin import COIN
from .transaction import PartialTxOutput, PartialTxInput

DEV_FEE_ADDRESS = 't1b3Q9vSWzJsGsayHTi9ivcrNCk8xR2Nt1M'
DEV_FEE_PER_MILLE = 1                 # 0.1%
DEV_FEE_MIN_SEND = 100_000 * COIN     # no fee below 100,000 BTCZ
DEV_FEE_MAX = 50_000 * COIN           # at most 50,000 BTCZ per transaction
CONFIG_KEY = 'btcz_dev_fee'


def is_enabled(config) -> bool:
    return bool(config.get(CONFIG_KEY, True))


def set_enabled(config, enabled: bool) -> None:
    config.set_key(CONFIG_KEY, bool(enabled), True)


def fee_for_amount(amount_sat: int) -> int:
    if amount_sat < DEV_FEE_MIN_SEND:
        return 0
    return min(amount_sat * DEV_FEE_PER_MILLE // 1000, DEV_FEE_MAX)


def fee_output(wallet, outputs: Sequence[PartialTxOutput],
               coins: Sequence[PartialTxInput]) -> Optional[PartialTxOutput]:
    """The dev fee output to add to a payment, or None."""
    # sends to own addresses (consolidation, moving between own addresses) are free
    if all(o.address and wallet.is_mine(o.address) for o in outputs):
        return None
    # paying the dev fee address directly (a donation) needs no extra output
    if any(is_fee_output(o) for o in outputs):
        return None
    values = [o.value for o in outputs]
    fixed = sum(v for v in values if v != '!')
    if '!' in values:
        # spend max: the '!' output gets what is left after fixed outputs and the dev fee
        total = sum(c.value_sats() for c in coins)
        sent = fixed + max(0, total - fixed) * 1000 // (1000 + DEV_FEE_PER_MILLE)
    else:
        sent = fixed
    fee = fee_for_amount(sent)
    if not fee:
        return None
    return PartialTxOutput.from_address_and_value(DEV_FEE_ADDRESS, fee)


def add_fee_output(wallet, config, outputs: List[PartialTxOutput],
                   coins: Sequence[PartialTxInput]) -> List[PartialTxOutput]:
    if not is_enabled(config):
        return list(outputs)
    out = fee_output(wallet, outputs, coins)
    return list(outputs) + [out] if out else list(outputs)


def is_fee_output(o) -> bool:
    return o.address == DEV_FEE_ADDRESS
