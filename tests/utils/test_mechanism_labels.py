"""
Tests for deriving three-class mechanism labels from variants.

Three perturbation modules previously each carried their own copy of this
derivation, and every copy substituted LOF — the majority class — for a variant
with no mechanism. These helpers raise instead, so a missing annotation stops
the run rather than entering the results as a real observation.

Covers:
- mechanism_label: an existing label_3class is returned unchanged
- mechanism_label: haploinsufficiency and autosomal-recessive both map to LOF
- mechanism_label: any other mechanism is its own label
- mechanism_label: a missing mechanism raises, naming the gene
- mechanism_label: a blank or null mechanism raises rather than becoming LOF
- gene_mechanism_labels: one label per gene, genes sorted and row-aligned
- gene_mechanism_labels: gene symbols are upper-cased before grouping
- gene_mechanism_labels: the majority label across a gene's variants wins
- gene_mechanism_labels: a tied gene raises rather than picking by dict order
- gene_mechanism_labels: a variant with no mechanism raises
- gene_mechanism_labels: an empty variant list yields empty, aligned arrays
"""

import numpy as np
import pytest

from esm2_mech.utils.constants import DN, GOF, LOF
from esm2_mech.utils.data import gene_mechanism_labels, mechanism_label


def _variant(gene="BRCA1", mechanism=GOF, **extra):
    return {"gene": gene, "mechanism": mechanism, **extra}


# ---------------------------------------------------------------------------
# mechanism_label
# ---------------------------------------------------------------------------


def test_an_existing_label_is_returned_unchanged():
    variant = _variant(mechanism="HI", label_3class=DN)
    assert mechanism_label(variant) == DN


@pytest.mark.parametrize("mechanism", ["HI", "AR"])
def test_loss_of_function_mechanisms_map_to_lof(mechanism):
    assert mechanism_label(_variant(mechanism=mechanism)) == LOF


@pytest.mark.parametrize("mechanism", [GOF, DN, LOF])
def test_other_mechanisms_are_their_own_label(mechanism):
    assert mechanism_label(_variant(mechanism=mechanism)) == mechanism


def test_a_missing_mechanism_raises_naming_the_gene():
    with pytest.raises(ValueError, match="TP53"):
        mechanism_label({"gene": "TP53"})


@pytest.mark.parametrize("mechanism", ["", None])
def test_a_blank_mechanism_raises_rather_than_becoming_lof(mechanism):
    """The former default silently filed unannotated variants under the
    majority class, which would inflate any probe that predicts it."""
    with pytest.raises(ValueError, match="no mechanism"):
        mechanism_label(_variant(mechanism=mechanism))


# ---------------------------------------------------------------------------
# gene_mechanism_labels
# ---------------------------------------------------------------------------


def test_one_label_per_gene_sorted_and_row_aligned():
    variants = [_variant("TP53", DN), _variant("BRCA1", GOF)]
    genes, labels = gene_mechanism_labels(variants)
    assert genes.tolist() == ["BRCA1", "TP53"]
    assert labels.tolist() == [GOF, DN]
    assert len(genes) == len(labels)


def test_gene_symbols_are_upper_cased_before_grouping():
    variants = [_variant("brca1", GOF), _variant("BRCA1", GOF)]
    genes, labels = gene_mechanism_labels(variants)
    assert genes.tolist() == ["BRCA1"]
    assert labels.tolist() == [GOF]


def test_the_majority_label_across_a_genes_variants_wins():
    variants = [
        _variant("BRCA1", GOF),
        _variant("BRCA1", GOF),
        _variant("BRCA1", DN),
    ]
    _genes, labels = gene_mechanism_labels(variants)
    assert labels.tolist() == [GOF]


def test_a_tied_gene_raises_rather_than_picking_by_dict_order():
    """Resolving a tie by insertion order makes the label depend on file order."""
    variants = [_variant("BRCA1", GOF), _variant("BRCA1", DN)]
    with pytest.raises(ValueError, match="no majority mechanism label"):
        gene_mechanism_labels(variants)


def test_a_variant_with_no_mechanism_raises():
    variants = [_variant("BRCA1", GOF), {"gene": "TP53"}]
    with pytest.raises(ValueError, match="no mechanism"):
        gene_mechanism_labels(variants)


def test_no_variants_yields_empty_aligned_arrays():
    genes, labels = gene_mechanism_labels([])
    assert len(genes) == len(labels) == 0
