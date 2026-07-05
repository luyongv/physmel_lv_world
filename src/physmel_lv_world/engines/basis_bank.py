from __future__ import annotations

import torch


def lv_basis_0(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 1.0 * scale)
    b = torch.cos(immune / 1.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_1(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 2.0 * scale)
    b = torch.cos(immune / 2.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_2(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 3.0 * scale)
    b = torch.cos(immune / 3.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_3(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 4.0 * scale)
    b = torch.cos(immune / 4.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_4(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 5.0 * scale)
    b = torch.cos(immune / 5.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_5(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 6.0 * scale)
    b = torch.cos(immune / 6.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_6(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 7.0 * scale)
    b = torch.cos(immune / 7.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_7(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 8.0 * scale)
    b = torch.cos(immune / 8.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_8(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 9.0 * scale)
    b = torch.cos(immune / 9.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_9(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 10.0 * scale)
    b = torch.cos(immune / 10.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_10(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 11.0 * scale)
    b = torch.cos(immune / 11.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_11(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 12.0 * scale)
    b = torch.cos(immune / 12.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_12(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 13.0 * scale)
    b = torch.cos(immune / 13.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_13(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 14.0 * scale)
    b = torch.cos(immune / 14.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_14(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 15.0 * scale)
    b = torch.cos(immune / 15.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_15(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 16.0 * scale)
    b = torch.cos(immune / 16.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_16(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 17.0 * scale)
    b = torch.cos(immune / 17.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_17(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 18.0 * scale)
    b = torch.cos(immune / 18.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_18(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 19.0 * scale)
    b = torch.cos(immune / 19.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_19(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 20.0 * scale)
    b = torch.cos(immune / 20.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_20(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 21.0 * scale)
    b = torch.cos(immune / 21.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_21(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 22.0 * scale)
    b = torch.cos(immune / 22.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_22(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 23.0 * scale)
    b = torch.cos(immune / 23.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_23(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 24.0 * scale)
    b = torch.cos(immune / 24.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_24(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 25.0 * scale)
    b = torch.cos(immune / 25.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_25(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 26.0 * scale)
    b = torch.cos(immune / 26.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_26(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 27.0 * scale)
    b = torch.cos(immune / 27.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_27(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 28.0 * scale)
    b = torch.cos(immune / 28.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_28(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 29.0 * scale)
    b = torch.cos(immune / 29.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_29(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 30.0 * scale)
    b = torch.cos(immune / 30.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_30(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 31.0 * scale)
    b = torch.cos(immune / 31.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_31(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 32.0 * scale)
    b = torch.cos(immune / 32.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_32(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 33.0 * scale)
    b = torch.cos(immune / 33.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_33(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 34.0 * scale)
    b = torch.cos(immune / 34.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_34(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 35.0 * scale)
    b = torch.cos(immune / 35.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_35(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 36.0 * scale)
    b = torch.cos(immune / 36.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_36(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 37.0 * scale)
    b = torch.cos(immune / 37.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_37(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 38.0 * scale)
    b = torch.cos(immune / 38.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_38(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 39.0 * scale)
    b = torch.cos(immune / 39.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_39(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 40.0 * scale)
    b = torch.cos(immune / 40.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_40(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 41.0 * scale)
    b = torch.cos(immune / 41.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_41(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 42.0 * scale)
    b = torch.cos(immune / 42.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_42(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 43.0 * scale)
    b = torch.cos(immune / 43.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_43(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 44.0 * scale)
    b = torch.cos(immune / 44.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_44(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 45.0 * scale)
    b = torch.cos(immune / 45.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_45(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 46.0 * scale)
    b = torch.cos(immune / 46.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_46(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 47.0 * scale)
    b = torch.cos(immune / 47.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_47(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 48.0 * scale)
    b = torch.cos(immune / 48.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_48(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 49.0 * scale)
    b = torch.cos(immune / 49.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_49(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 50.0 * scale)
    b = torch.cos(immune / 50.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_50(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 51.0 * scale)
    b = torch.cos(immune / 51.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_51(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 52.0 * scale)
    b = torch.cos(immune / 52.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_52(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 53.0 * scale)
    b = torch.cos(immune / 53.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_53(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 54.0 * scale)
    b = torch.cos(immune / 54.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_54(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 55.0 * scale)
    b = torch.cos(immune / 55.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_55(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 56.0 * scale)
    b = torch.cos(immune / 56.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_56(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 57.0 * scale)
    b = torch.cos(immune / 57.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_57(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 58.0 * scale)
    b = torch.cos(immune / 58.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_58(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 59.0 * scale)
    b = torch.cos(immune / 59.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_59(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 60.0 * scale)
    b = torch.cos(immune / 60.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_60(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 61.0 * scale)
    b = torch.cos(immune / 61.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_61(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 62.0 * scale)
    b = torch.cos(immune / 62.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_62(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 63.0 * scale)
    b = torch.cos(immune / 63.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_63(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 64.0 * scale)
    b = torch.cos(immune / 64.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_64(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 65.0 * scale)
    b = torch.cos(immune / 65.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_65(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 66.0 * scale)
    b = torch.cos(immune / 66.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_66(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 67.0 * scale)
    b = torch.cos(immune / 67.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_67(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 68.0 * scale)
    b = torch.cos(immune / 68.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_68(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 69.0 * scale)
    b = torch.cos(immune / 69.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_69(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 70.0 * scale)
    b = torch.cos(immune / 70.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_70(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 71.0 * scale)
    b = torch.cos(immune / 71.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_71(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 72.0 * scale)
    b = torch.cos(immune / 72.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_72(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 73.0 * scale)
    b = torch.cos(immune / 73.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_73(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 74.0 * scale)
    b = torch.cos(immune / 74.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_74(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 75.0 * scale)
    b = torch.cos(immune / 75.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_75(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 76.0 * scale)
    b = torch.cos(immune / 76.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_76(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 77.0 * scale)
    b = torch.cos(immune / 77.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_77(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 78.0 * scale)
    b = torch.cos(immune / 78.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_78(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 79.0 * scale)
    b = torch.cos(immune / 79.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_79(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 80.0 * scale)
    b = torch.cos(immune / 80.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_80(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 81.0 * scale)
    b = torch.cos(immune / 81.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_81(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 82.0 * scale)
    b = torch.cos(immune / 82.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_82(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 83.0 * scale)
    b = torch.cos(immune / 83.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_83(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 84.0 * scale)
    b = torch.cos(immune / 84.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_84(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 85.0 * scale)
    b = torch.cos(immune / 85.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_85(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 86.0 * scale)
    b = torch.cos(immune / 86.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_86(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 87.0 * scale)
    b = torch.cos(immune / 87.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_87(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 88.0 * scale)
    b = torch.cos(immune / 88.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_88(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 89.0 * scale)
    b = torch.cos(immune / 89.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_89(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 90.0 * scale)
    b = torch.cos(immune / 90.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_90(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 91.0 * scale)
    b = torch.cos(immune / 91.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_91(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 92.0 * scale)
    b = torch.cos(immune / 92.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_92(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 93.0 * scale)
    b = torch.cos(immune / 93.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_93(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 94.0 * scale)
    b = torch.cos(immune / 94.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_94(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 95.0 * scale)
    b = torch.cos(immune / 95.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_95(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 96.0 * scale)
    b = torch.cos(immune / 96.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_96(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 97.0 * scale)
    b = torch.cos(immune / 97.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_97(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 98.0 * scale)
    b = torch.cos(immune / 98.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_98(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 99.0 * scale)
    b = torch.cos(immune / 99.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_99(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 100.0 * scale)
    b = torch.cos(immune / 100.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_100(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 101.0 * scale)
    b = torch.cos(immune / 101.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_101(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 102.0 * scale)
    b = torch.cos(immune / 102.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_102(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 103.0 * scale)
    b = torch.cos(immune / 103.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_103(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 104.0 * scale)
    b = torch.cos(immune / 104.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_104(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 105.0 * scale)
    b = torch.cos(immune / 105.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_105(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 106.0 * scale)
    b = torch.cos(immune / 106.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_106(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 107.0 * scale)
    b = torch.cos(immune / 107.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_107(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 108.0 * scale)
    b = torch.cos(immune / 108.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_108(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 109.0 * scale)
    b = torch.cos(immune / 109.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_109(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 110.0 * scale)
    b = torch.cos(immune / 110.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_110(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 111.0 * scale)
    b = torch.cos(immune / 111.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_111(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 112.0 * scale)
    b = torch.cos(immune / 112.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_112(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 113.0 * scale)
    b = torch.cos(immune / 113.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_113(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 114.0 * scale)
    b = torch.cos(immune / 114.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_114(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 115.0 * scale)
    b = torch.cos(immune / 115.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_115(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 116.0 * scale)
    b = torch.cos(immune / 116.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_116(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 117.0 * scale)
    b = torch.cos(immune / 117.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_117(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 118.0 * scale)
    b = torch.cos(immune / 118.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_118(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 119.0 * scale)
    b = torch.cos(immune / 119.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_119(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 120.0 * scale)
    b = torch.cos(immune / 120.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_120(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 121.0 * scale)
    b = torch.cos(immune / 121.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_121(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 122.0 * scale)
    b = torch.cos(immune / 122.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_122(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 123.0 * scale)
    b = torch.cos(immune / 123.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_123(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 124.0 * scale)
    b = torch.cos(immune / 124.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_124(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 125.0 * scale)
    b = torch.cos(immune / 125.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_125(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 126.0 * scale)
    b = torch.cos(immune / 126.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_126(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 127.0 * scale)
    b = torch.cos(immune / 127.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_127(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 128.0 * scale)
    b = torch.cos(immune / 128.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_128(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 129.0 * scale)
    b = torch.cos(immune / 129.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_129(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 130.0 * scale)
    b = torch.cos(immune / 130.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_130(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 131.0 * scale)
    b = torch.cos(immune / 131.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_131(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 132.0 * scale)
    b = torch.cos(immune / 132.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_132(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 133.0 * scale)
    b = torch.cos(immune / 133.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_133(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 134.0 * scale)
    b = torch.cos(immune / 134.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_134(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 135.0 * scale)
    b = torch.cos(immune / 135.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_135(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 136.0 * scale)
    b = torch.cos(immune / 136.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_136(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 137.0 * scale)
    b = torch.cos(immune / 137.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_137(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 138.0 * scale)
    b = torch.cos(immune / 138.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_138(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 139.0 * scale)
    b = torch.cos(immune / 139.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_139(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 140.0 * scale)
    b = torch.cos(immune / 140.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_140(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 141.0 * scale)
    b = torch.cos(immune / 141.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_141(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 142.0 * scale)
    b = torch.cos(immune / 142.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_142(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 143.0 * scale)
    b = torch.cos(immune / 143.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_143(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 144.0 * scale)
    b = torch.cos(immune / 144.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_144(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 145.0 * scale)
    b = torch.cos(immune / 145.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_145(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 146.0 * scale)
    b = torch.cos(immune / 146.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_146(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 147.0 * scale)
    b = torch.cos(immune / 147.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_147(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 148.0 * scale)
    b = torch.cos(immune / 148.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_148(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 149.0 * scale)
    b = torch.cos(immune / 149.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_149(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 150.0 * scale)
    b = torch.cos(immune / 150.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_150(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 151.0 * scale)
    b = torch.cos(immune / 151.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_151(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 152.0 * scale)
    b = torch.cos(immune / 152.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_152(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 153.0 * scale)
    b = torch.cos(immune / 153.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_153(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 154.0 * scale)
    b = torch.cos(immune / 154.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_154(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 155.0 * scale)
    b = torch.cos(immune / 155.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_155(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 156.0 * scale)
    b = torch.cos(immune / 156.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_156(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 157.0 * scale)
    b = torch.cos(immune / 157.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_157(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 158.0 * scale)
    b = torch.cos(immune / 158.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_158(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 159.0 * scale)
    b = torch.cos(immune / 159.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


def lv_basis_159(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tumor = state[..., 0]
    immune = state[..., 1]
    a = torch.sin(tumor * 160.0 * scale)
    b = torch.cos(immune / 160.0 + scale)
    c = torch.tanh(a + b)
    return torch.stack([a + c, b - c], dim=-1)


BASIS_COUNT = 160


def basis_bank(state: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    values = [
        lv_basis_0(state, scale),
        lv_basis_1(state, scale),
        lv_basis_2(state, scale),
        lv_basis_3(state, scale),
        lv_basis_4(state, scale),
        lv_basis_5(state, scale),
        lv_basis_6(state, scale),
        lv_basis_7(state, scale),
        lv_basis_8(state, scale),
        lv_basis_9(state, scale),
        lv_basis_10(state, scale),
        lv_basis_11(state, scale),
        lv_basis_12(state, scale),
        lv_basis_13(state, scale),
        lv_basis_14(state, scale),
        lv_basis_15(state, scale),
        lv_basis_16(state, scale),
        lv_basis_17(state, scale),
        lv_basis_18(state, scale),
        lv_basis_19(state, scale),
        lv_basis_20(state, scale),
        lv_basis_21(state, scale),
        lv_basis_22(state, scale),
        lv_basis_23(state, scale),
        lv_basis_24(state, scale),
        lv_basis_25(state, scale),
        lv_basis_26(state, scale),
        lv_basis_27(state, scale),
        lv_basis_28(state, scale),
        lv_basis_29(state, scale),
        lv_basis_30(state, scale),
        lv_basis_31(state, scale),
        lv_basis_32(state, scale),
        lv_basis_33(state, scale),
        lv_basis_34(state, scale),
        lv_basis_35(state, scale),
        lv_basis_36(state, scale),
        lv_basis_37(state, scale),
        lv_basis_38(state, scale),
        lv_basis_39(state, scale),
        lv_basis_40(state, scale),
        lv_basis_41(state, scale),
        lv_basis_42(state, scale),
        lv_basis_43(state, scale),
        lv_basis_44(state, scale),
        lv_basis_45(state, scale),
        lv_basis_46(state, scale),
        lv_basis_47(state, scale),
        lv_basis_48(state, scale),
        lv_basis_49(state, scale),
        lv_basis_50(state, scale),
        lv_basis_51(state, scale),
        lv_basis_52(state, scale),
        lv_basis_53(state, scale),
        lv_basis_54(state, scale),
        lv_basis_55(state, scale),
        lv_basis_56(state, scale),
        lv_basis_57(state, scale),
        lv_basis_58(state, scale),
        lv_basis_59(state, scale),
        lv_basis_60(state, scale),
        lv_basis_61(state, scale),
        lv_basis_62(state, scale),
        lv_basis_63(state, scale),
        lv_basis_64(state, scale),
        lv_basis_65(state, scale),
        lv_basis_66(state, scale),
        lv_basis_67(state, scale),
        lv_basis_68(state, scale),
        lv_basis_69(state, scale),
        lv_basis_70(state, scale),
        lv_basis_71(state, scale),
        lv_basis_72(state, scale),
        lv_basis_73(state, scale),
        lv_basis_74(state, scale),
        lv_basis_75(state, scale),
        lv_basis_76(state, scale),
        lv_basis_77(state, scale),
        lv_basis_78(state, scale),
        lv_basis_79(state, scale),
        lv_basis_80(state, scale),
        lv_basis_81(state, scale),
        lv_basis_82(state, scale),
        lv_basis_83(state, scale),
        lv_basis_84(state, scale),
        lv_basis_85(state, scale),
        lv_basis_86(state, scale),
        lv_basis_87(state, scale),
        lv_basis_88(state, scale),
        lv_basis_89(state, scale),
        lv_basis_90(state, scale),
        lv_basis_91(state, scale),
        lv_basis_92(state, scale),
        lv_basis_93(state, scale),
        lv_basis_94(state, scale),
        lv_basis_95(state, scale),
        lv_basis_96(state, scale),
        lv_basis_97(state, scale),
        lv_basis_98(state, scale),
        lv_basis_99(state, scale),
        lv_basis_100(state, scale),
        lv_basis_101(state, scale),
        lv_basis_102(state, scale),
        lv_basis_103(state, scale),
        lv_basis_104(state, scale),
        lv_basis_105(state, scale),
        lv_basis_106(state, scale),
        lv_basis_107(state, scale),
        lv_basis_108(state, scale),
        lv_basis_109(state, scale),
        lv_basis_110(state, scale),
        lv_basis_111(state, scale),
        lv_basis_112(state, scale),
        lv_basis_113(state, scale),
        lv_basis_114(state, scale),
        lv_basis_115(state, scale),
        lv_basis_116(state, scale),
        lv_basis_117(state, scale),
        lv_basis_118(state, scale),
        lv_basis_119(state, scale),
        lv_basis_120(state, scale),
        lv_basis_121(state, scale),
        lv_basis_122(state, scale),
        lv_basis_123(state, scale),
        lv_basis_124(state, scale),
        lv_basis_125(state, scale),
        lv_basis_126(state, scale),
        lv_basis_127(state, scale),
        lv_basis_128(state, scale),
        lv_basis_129(state, scale),
        lv_basis_130(state, scale),
        lv_basis_131(state, scale),
        lv_basis_132(state, scale),
        lv_basis_133(state, scale),
        lv_basis_134(state, scale),
        lv_basis_135(state, scale),
        lv_basis_136(state, scale),
        lv_basis_137(state, scale),
        lv_basis_138(state, scale),
        lv_basis_139(state, scale),
        lv_basis_140(state, scale),
        lv_basis_141(state, scale),
        lv_basis_142(state, scale),
        lv_basis_143(state, scale),
        lv_basis_144(state, scale),
        lv_basis_145(state, scale),
        lv_basis_146(state, scale),
        lv_basis_147(state, scale),
        lv_basis_148(state, scale),
        lv_basis_149(state, scale),
        lv_basis_150(state, scale),
        lv_basis_151(state, scale),
        lv_basis_152(state, scale),
        lv_basis_153(state, scale),
        lv_basis_154(state, scale),
        lv_basis_155(state, scale),
        lv_basis_156(state, scale),
        lv_basis_157(state, scale),
        lv_basis_158(state, scale),
        lv_basis_159(state, scale),
    ]
    return torch.stack(values, dim=-2)
