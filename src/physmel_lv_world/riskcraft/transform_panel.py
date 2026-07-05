from __future__ import annotations

import torch


def survival_transform_0(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 2.0)
    decay = torch.exp(-time.clamp_min(0.0) / 366.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_1(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 3.0)
    decay = torch.exp(-time.clamp_min(0.0) / 367.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_2(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 4.0)
    decay = torch.exp(-time.clamp_min(0.0) / 368.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_3(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 5.0)
    decay = torch.exp(-time.clamp_min(0.0) / 369.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_4(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 6.0)
    decay = torch.exp(-time.clamp_min(0.0) / 370.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_5(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 7.0)
    decay = torch.exp(-time.clamp_min(0.0) / 371.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_6(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 8.0)
    decay = torch.exp(-time.clamp_min(0.0) / 372.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_7(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 9.0)
    decay = torch.exp(-time.clamp_min(0.0) / 373.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_8(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 10.0)
    decay = torch.exp(-time.clamp_min(0.0) / 374.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_9(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 11.0)
    decay = torch.exp(-time.clamp_min(0.0) / 375.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_10(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 12.0)
    decay = torch.exp(-time.clamp_min(0.0) / 376.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_11(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 13.0)
    decay = torch.exp(-time.clamp_min(0.0) / 377.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_12(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 14.0)
    decay = torch.exp(-time.clamp_min(0.0) / 378.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_13(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 15.0)
    decay = torch.exp(-time.clamp_min(0.0) / 379.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_14(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 16.0)
    decay = torch.exp(-time.clamp_min(0.0) / 380.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_15(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 17.0)
    decay = torch.exp(-time.clamp_min(0.0) / 381.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_16(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 18.0)
    decay = torch.exp(-time.clamp_min(0.0) / 382.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_17(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 19.0)
    decay = torch.exp(-time.clamp_min(0.0) / 383.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_18(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 20.0)
    decay = torch.exp(-time.clamp_min(0.0) / 384.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_19(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 21.0)
    decay = torch.exp(-time.clamp_min(0.0) / 385.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_20(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 22.0)
    decay = torch.exp(-time.clamp_min(0.0) / 386.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_21(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 23.0)
    decay = torch.exp(-time.clamp_min(0.0) / 387.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_22(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 24.0)
    decay = torch.exp(-time.clamp_min(0.0) / 388.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_23(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 25.0)
    decay = torch.exp(-time.clamp_min(0.0) / 389.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_24(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 26.0)
    decay = torch.exp(-time.clamp_min(0.0) / 390.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_25(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 27.0)
    decay = torch.exp(-time.clamp_min(0.0) / 391.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_26(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 28.0)
    decay = torch.exp(-time.clamp_min(0.0) / 392.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_27(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 29.0)
    decay = torch.exp(-time.clamp_min(0.0) / 393.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_28(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 30.0)
    decay = torch.exp(-time.clamp_min(0.0) / 394.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_29(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 31.0)
    decay = torch.exp(-time.clamp_min(0.0) / 395.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_30(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 32.0)
    decay = torch.exp(-time.clamp_min(0.0) / 396.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_31(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 33.0)
    decay = torch.exp(-time.clamp_min(0.0) / 397.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_32(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 34.0)
    decay = torch.exp(-time.clamp_min(0.0) / 398.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_33(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 35.0)
    decay = torch.exp(-time.clamp_min(0.0) / 399.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_34(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 36.0)
    decay = torch.exp(-time.clamp_min(0.0) / 400.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_35(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 37.0)
    decay = torch.exp(-time.clamp_min(0.0) / 401.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_36(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 38.0)
    decay = torch.exp(-time.clamp_min(0.0) / 402.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_37(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 39.0)
    decay = torch.exp(-time.clamp_min(0.0) / 403.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_38(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 40.0)
    decay = torch.exp(-time.clamp_min(0.0) / 404.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_39(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 41.0)
    decay = torch.exp(-time.clamp_min(0.0) / 405.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_40(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 42.0)
    decay = torch.exp(-time.clamp_min(0.0) / 406.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_41(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 43.0)
    decay = torch.exp(-time.clamp_min(0.0) / 407.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_42(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 44.0)
    decay = torch.exp(-time.clamp_min(0.0) / 408.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_43(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 45.0)
    decay = torch.exp(-time.clamp_min(0.0) / 409.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_44(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 46.0)
    decay = torch.exp(-time.clamp_min(0.0) / 410.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_45(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 47.0)
    decay = torch.exp(-time.clamp_min(0.0) / 411.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_46(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 48.0)
    decay = torch.exp(-time.clamp_min(0.0) / 412.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_47(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 49.0)
    decay = torch.exp(-time.clamp_min(0.0) / 413.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_48(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 50.0)
    decay = torch.exp(-time.clamp_min(0.0) / 414.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_49(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 51.0)
    decay = torch.exp(-time.clamp_min(0.0) / 415.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_50(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 52.0)
    decay = torch.exp(-time.clamp_min(0.0) / 416.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_51(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 53.0)
    decay = torch.exp(-time.clamp_min(0.0) / 417.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_52(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 54.0)
    decay = torch.exp(-time.clamp_min(0.0) / 418.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_53(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 55.0)
    decay = torch.exp(-time.clamp_min(0.0) / 419.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_54(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 56.0)
    decay = torch.exp(-time.clamp_min(0.0) / 420.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_55(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 57.0)
    decay = torch.exp(-time.clamp_min(0.0) / 421.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_56(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 58.0)
    decay = torch.exp(-time.clamp_min(0.0) / 422.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_57(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 59.0)
    decay = torch.exp(-time.clamp_min(0.0) / 423.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_58(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 60.0)
    decay = torch.exp(-time.clamp_min(0.0) / 424.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_59(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 61.0)
    decay = torch.exp(-time.clamp_min(0.0) / 425.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_60(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 62.0)
    decay = torch.exp(-time.clamp_min(0.0) / 426.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_61(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 63.0)
    decay = torch.exp(-time.clamp_min(0.0) / 427.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_62(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 64.0)
    decay = torch.exp(-time.clamp_min(0.0) / 428.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_63(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 65.0)
    decay = torch.exp(-time.clamp_min(0.0) / 429.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_64(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 66.0)
    decay = torch.exp(-time.clamp_min(0.0) / 430.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_65(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 67.0)
    decay = torch.exp(-time.clamp_min(0.0) / 431.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_66(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 68.0)
    decay = torch.exp(-time.clamp_min(0.0) / 432.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_67(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 69.0)
    decay = torch.exp(-time.clamp_min(0.0) / 433.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_68(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 70.0)
    decay = torch.exp(-time.clamp_min(0.0) / 434.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_69(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 71.0)
    decay = torch.exp(-time.clamp_min(0.0) / 435.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_70(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 72.0)
    decay = torch.exp(-time.clamp_min(0.0) / 436.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_71(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 73.0)
    decay = torch.exp(-time.clamp_min(0.0) / 437.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_72(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 74.0)
    decay = torch.exp(-time.clamp_min(0.0) / 438.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_73(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 75.0)
    decay = torch.exp(-time.clamp_min(0.0) / 439.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_74(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 76.0)
    decay = torch.exp(-time.clamp_min(0.0) / 440.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_75(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 77.0)
    decay = torch.exp(-time.clamp_min(0.0) / 441.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_76(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 78.0)
    decay = torch.exp(-time.clamp_min(0.0) / 442.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_77(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 79.0)
    decay = torch.exp(-time.clamp_min(0.0) / 443.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_78(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 80.0)
    decay = torch.exp(-time.clamp_min(0.0) / 444.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_79(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 81.0)
    decay = torch.exp(-time.clamp_min(0.0) / 445.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_80(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 82.0)
    decay = torch.exp(-time.clamp_min(0.0) / 446.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_81(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 83.0)
    decay = torch.exp(-time.clamp_min(0.0) / 447.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_82(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 84.0)
    decay = torch.exp(-time.clamp_min(0.0) / 448.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_83(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 85.0)
    decay = torch.exp(-time.clamp_min(0.0) / 449.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_84(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 86.0)
    decay = torch.exp(-time.clamp_min(0.0) / 450.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_85(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 87.0)
    decay = torch.exp(-time.clamp_min(0.0) / 451.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_86(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 88.0)
    decay = torch.exp(-time.clamp_min(0.0) / 452.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_87(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 89.0)
    decay = torch.exp(-time.clamp_min(0.0) / 453.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_88(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 90.0)
    decay = torch.exp(-time.clamp_min(0.0) / 454.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_89(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 91.0)
    decay = torch.exp(-time.clamp_min(0.0) / 455.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_90(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 92.0)
    decay = torch.exp(-time.clamp_min(0.0) / 456.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_91(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 93.0)
    decay = torch.exp(-time.clamp_min(0.0) / 457.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_92(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 94.0)
    decay = torch.exp(-time.clamp_min(0.0) / 458.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_93(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 95.0)
    decay = torch.exp(-time.clamp_min(0.0) / 459.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_94(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 96.0)
    decay = torch.exp(-time.clamp_min(0.0) / 460.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_95(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 97.0)
    decay = torch.exp(-time.clamp_min(0.0) / 461.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_96(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 98.0)
    decay = torch.exp(-time.clamp_min(0.0) / 462.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_97(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 99.0)
    decay = torch.exp(-time.clamp_min(0.0) / 463.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_98(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 100.0)
    decay = torch.exp(-time.clamp_min(0.0) / 464.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_99(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 101.0)
    decay = torch.exp(-time.clamp_min(0.0) / 465.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_100(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 102.0)
    decay = torch.exp(-time.clamp_min(0.0) / 466.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_101(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 103.0)
    decay = torch.exp(-time.clamp_min(0.0) / 467.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_102(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 104.0)
    decay = torch.exp(-time.clamp_min(0.0) / 468.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_103(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 105.0)
    decay = torch.exp(-time.clamp_min(0.0) / 469.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_104(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 106.0)
    decay = torch.exp(-time.clamp_min(0.0) / 470.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_105(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 107.0)
    decay = torch.exp(-time.clamp_min(0.0) / 471.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_106(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 108.0)
    decay = torch.exp(-time.clamp_min(0.0) / 472.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_107(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 109.0)
    decay = torch.exp(-time.clamp_min(0.0) / 473.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_108(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 110.0)
    decay = torch.exp(-time.clamp_min(0.0) / 474.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_109(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 111.0)
    decay = torch.exp(-time.clamp_min(0.0) / 475.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_110(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 112.0)
    decay = torch.exp(-time.clamp_min(0.0) / 476.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_111(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 113.0)
    decay = torch.exp(-time.clamp_min(0.0) / 477.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_112(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 114.0)
    decay = torch.exp(-time.clamp_min(0.0) / 478.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_113(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 115.0)
    decay = torch.exp(-time.clamp_min(0.0) / 479.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_114(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 116.0)
    decay = torch.exp(-time.clamp_min(0.0) / 480.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_115(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 117.0)
    decay = torch.exp(-time.clamp_min(0.0) / 481.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_116(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 118.0)
    decay = torch.exp(-time.clamp_min(0.0) / 482.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_117(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 119.0)
    decay = torch.exp(-time.clamp_min(0.0) / 483.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_118(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 120.0)
    decay = torch.exp(-time.clamp_min(0.0) / 484.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_119(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 121.0)
    decay = torch.exp(-time.clamp_min(0.0) / 485.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_120(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 122.0)
    decay = torch.exp(-time.clamp_min(0.0) / 486.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_121(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 123.0)
    decay = torch.exp(-time.clamp_min(0.0) / 487.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_122(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 124.0)
    decay = torch.exp(-time.clamp_min(0.0) / 488.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_123(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 125.0)
    decay = torch.exp(-time.clamp_min(0.0) / 489.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_124(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 126.0)
    decay = torch.exp(-time.clamp_min(0.0) / 490.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_125(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 127.0)
    decay = torch.exp(-time.clamp_min(0.0) / 491.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_126(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 128.0)
    decay = torch.exp(-time.clamp_min(0.0) / 492.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_127(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 129.0)
    decay = torch.exp(-time.clamp_min(0.0) / 493.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_128(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 130.0)
    decay = torch.exp(-time.clamp_min(0.0) / 494.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_129(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 131.0)
    decay = torch.exp(-time.clamp_min(0.0) / 495.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_130(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 132.0)
    decay = torch.exp(-time.clamp_min(0.0) / 496.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_131(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 133.0)
    decay = torch.exp(-time.clamp_min(0.0) / 497.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_132(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 134.0)
    decay = torch.exp(-time.clamp_min(0.0) / 498.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_133(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 135.0)
    decay = torch.exp(-time.clamp_min(0.0) / 499.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_134(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 136.0)
    decay = torch.exp(-time.clamp_min(0.0) / 500.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_135(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 137.0)
    decay = torch.exp(-time.clamp_min(0.0) / 501.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_136(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 138.0)
    decay = torch.exp(-time.clamp_min(0.0) / 502.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_137(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 139.0)
    decay = torch.exp(-time.clamp_min(0.0) / 503.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_138(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 140.0)
    decay = torch.exp(-time.clamp_min(0.0) / 504.0)
    centered = weight - weight.mean()
    return centered * decay


def survival_transform_139(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    weight = torch.sigmoid(risk / 141.0)
    decay = torch.exp(-time.clamp_min(0.0) / 505.0)
    centered = weight - weight.mean()
    return centered * decay


def transform_panel(risk: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    values = [
        survival_transform_0(risk, time),
        survival_transform_1(risk, time),
        survival_transform_2(risk, time),
        survival_transform_3(risk, time),
        survival_transform_4(risk, time),
        survival_transform_5(risk, time),
        survival_transform_6(risk, time),
        survival_transform_7(risk, time),
        survival_transform_8(risk, time),
        survival_transform_9(risk, time),
        survival_transform_10(risk, time),
        survival_transform_11(risk, time),
        survival_transform_12(risk, time),
        survival_transform_13(risk, time),
        survival_transform_14(risk, time),
        survival_transform_15(risk, time),
        survival_transform_16(risk, time),
        survival_transform_17(risk, time),
        survival_transform_18(risk, time),
        survival_transform_19(risk, time),
        survival_transform_20(risk, time),
        survival_transform_21(risk, time),
        survival_transform_22(risk, time),
        survival_transform_23(risk, time),
        survival_transform_24(risk, time),
        survival_transform_25(risk, time),
        survival_transform_26(risk, time),
        survival_transform_27(risk, time),
        survival_transform_28(risk, time),
        survival_transform_29(risk, time),
        survival_transform_30(risk, time),
        survival_transform_31(risk, time),
        survival_transform_32(risk, time),
        survival_transform_33(risk, time),
        survival_transform_34(risk, time),
        survival_transform_35(risk, time),
        survival_transform_36(risk, time),
        survival_transform_37(risk, time),
        survival_transform_38(risk, time),
        survival_transform_39(risk, time),
        survival_transform_40(risk, time),
        survival_transform_41(risk, time),
        survival_transform_42(risk, time),
        survival_transform_43(risk, time),
        survival_transform_44(risk, time),
        survival_transform_45(risk, time),
        survival_transform_46(risk, time),
        survival_transform_47(risk, time),
        survival_transform_48(risk, time),
        survival_transform_49(risk, time),
        survival_transform_50(risk, time),
        survival_transform_51(risk, time),
        survival_transform_52(risk, time),
        survival_transform_53(risk, time),
        survival_transform_54(risk, time),
        survival_transform_55(risk, time),
        survival_transform_56(risk, time),
        survival_transform_57(risk, time),
        survival_transform_58(risk, time),
        survival_transform_59(risk, time),
        survival_transform_60(risk, time),
        survival_transform_61(risk, time),
        survival_transform_62(risk, time),
        survival_transform_63(risk, time),
        survival_transform_64(risk, time),
        survival_transform_65(risk, time),
        survival_transform_66(risk, time),
        survival_transform_67(risk, time),
        survival_transform_68(risk, time),
        survival_transform_69(risk, time),
        survival_transform_70(risk, time),
        survival_transform_71(risk, time),
        survival_transform_72(risk, time),
        survival_transform_73(risk, time),
        survival_transform_74(risk, time),
        survival_transform_75(risk, time),
        survival_transform_76(risk, time),
        survival_transform_77(risk, time),
        survival_transform_78(risk, time),
        survival_transform_79(risk, time),
        survival_transform_80(risk, time),
        survival_transform_81(risk, time),
        survival_transform_82(risk, time),
        survival_transform_83(risk, time),
        survival_transform_84(risk, time),
        survival_transform_85(risk, time),
        survival_transform_86(risk, time),
        survival_transform_87(risk, time),
        survival_transform_88(risk, time),
        survival_transform_89(risk, time),
        survival_transform_90(risk, time),
        survival_transform_91(risk, time),
        survival_transform_92(risk, time),
        survival_transform_93(risk, time),
        survival_transform_94(risk, time),
        survival_transform_95(risk, time),
        survival_transform_96(risk, time),
        survival_transform_97(risk, time),
        survival_transform_98(risk, time),
        survival_transform_99(risk, time),
        survival_transform_100(risk, time),
        survival_transform_101(risk, time),
        survival_transform_102(risk, time),
        survival_transform_103(risk, time),
        survival_transform_104(risk, time),
        survival_transform_105(risk, time),
        survival_transform_106(risk, time),
        survival_transform_107(risk, time),
        survival_transform_108(risk, time),
        survival_transform_109(risk, time),
        survival_transform_110(risk, time),
        survival_transform_111(risk, time),
        survival_transform_112(risk, time),
        survival_transform_113(risk, time),
        survival_transform_114(risk, time),
        survival_transform_115(risk, time),
        survival_transform_116(risk, time),
        survival_transform_117(risk, time),
        survival_transform_118(risk, time),
        survival_transform_119(risk, time),
        survival_transform_120(risk, time),
        survival_transform_121(risk, time),
        survival_transform_122(risk, time),
        survival_transform_123(risk, time),
        survival_transform_124(risk, time),
        survival_transform_125(risk, time),
        survival_transform_126(risk, time),
        survival_transform_127(risk, time),
        survival_transform_128(risk, time),
        survival_transform_129(risk, time),
        survival_transform_130(risk, time),
        survival_transform_131(risk, time),
        survival_transform_132(risk, time),
        survival_transform_133(risk, time),
        survival_transform_134(risk, time),
        survival_transform_135(risk, time),
        survival_transform_136(risk, time),
        survival_transform_137(risk, time),
        survival_transform_138(risk, time),
        survival_transform_139(risk, time),
    ]
    return torch.stack(values, dim=-1)
