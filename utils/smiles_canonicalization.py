#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility helpers for SMILES canonicalization and molecule IDs.

The original project expected these helpers to live in utils.smiles_canonicalization
but the file was missing in this checkout.  This implementation wraps RDKit so
the downstream metrics code can canonicalize SMILES strings and compare molecules
through a stable identifier (InChI when possible, canonical SMILES otherwise).
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

# Silence noisy RDKit logs for invalid inputs; metrics.py already handles errors.
RDLogger.DisableLog("rdApp.*")


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string into an RDKit Mol, returning None on failure."""
    if smiles is None:
        return None
    stripped = smiles.strip()
    if not stripped:
        return None
    try:
        mol = Chem.MolFromSmiles(stripped)
    except Exception:
        mol = None
    return mol


def canonicalize_molecule_smiles(
    smiles: Optional[str],
    return_none_for_error: bool = True,
) -> Optional[str]:
    """
    Convert the provided SMILES into RDKit's canonical, isomeric form.

    Parameters
    ----------
    smiles:
        Raw SMILES string.  Can include whitespace; empty/None will yield None.
    return_none_for_error:
        If True, return None when canonicalization fails; otherwise, return the
        original string to allow the caller to fall back on raw inputs.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None if return_none_for_error else smiles
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None if return_none_for_error else smiles


def get_molecule_id(
    smiles: Optional[str],
    remove_duplicate: bool = True,
) -> Optional[str]:
    """
    Generate a stable identifier for a molecule so metrics can compare equality.

    We try to use the InChI string (which uniquely encodes molecular structure).
    If RDKit fails to build an InChI, we fall back to canonical SMILES.

    Parameters
    ----------
    smiles:
        Raw SMILES string.
    remove_duplicate:
        Kept for API compatibility with the original project.  When False we
        simply return the identifier as-is; when True duplicates are implicitly
        removed because the identifier is deterministic.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None

    identifier: Optional[str]
    try:
        identifier = inchi.MolToInchi(mol)
    except Exception:
        identifier = None

    if not identifier:
        identifier = canonicalize_molecule_smiles(smiles, return_none_for_error=True)

    if identifier is None:
        return None

    if remove_duplicate:
        # Deterministic string already serves as a deduplicated identifier.
        return identifier
    return identifier

