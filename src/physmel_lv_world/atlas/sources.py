from __future__ import annotations

from physmel_lv_world.vault.types import DatasetSource


def known_sources() -> tuple[DatasetSource, ...]:
    return (
        DatasetSource(
            name="TCGA-SKCM",
            url="https://portal.gdc.cancer.gov/projects/TCGA-SKCM",
            version="GDC Data Release 40",
            license_or_policy="NIH Genomic Data Sharing Policy",
            access="public portal with controlled access elements",
        ),
        DatasetSource(
            name="GSE65904",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65904",
            version="GEO public series",
            license_or_policy="NCBI GEO public access terms",
            access="public",
        ),
        DatasetSource(
            name="GSE72056",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72056",
            version="GEO public series",
            license_or_policy="NCBI GEO public access terms",
            access="public",
        ),
    )


def source_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in known_sources():
        rows.append(
            {
                "name": source.name,
                "url": source.url,
                "version": source.version,
                "policy": source.license_or_policy,
                "access": source.access,
            }
        )
    return rows
