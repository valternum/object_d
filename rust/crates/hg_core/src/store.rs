//! Serialization of Elias–Fano sequences and hypergraphs into a container.

use std::borrow::Cow;

use crate::bits::RankSelect;
use crate::container::{Container, ContainerWriter, Payload};
use crate::ef::EliasFano;
use crate::error::{CoreError, Result};
use crate::graph::{Hypergraph, HypergraphParts};

/// Adds the sections of an Elias–Fano sequence under `name`
/// (`.m` metadata `[n, low_bits, high_len]`, `.lo`, `.hw`, `.hs`).
pub fn write_ef<'a>(w: &mut ContainerWriter<'a>, name: &str, ef: &'a EliasFano<'_>) -> Result<()> {
    w.add(
        &format!("{name}.m"),
        Payload::OwnedU64(vec![
            ef.len() as u64,
            u64::from(ef.low_bits()),
            ef.high().len() as u64,
        ]),
    )?;
    w.add(&format!("{name}.lo"), Payload::U64(ef.low_words()))?;
    w.add(&format!("{name}.hw"), Payload::U64(ef.high().words()))?;
    w.add(&format!("{name}.hs"), Payload::U64(ef.high().supers()))?;
    Ok(())
}

/// Reads an Elias–Fano sequence written by [`write_ef`].
pub fn read_ef<'c>(c: &'c Container, name: &str) -> Result<EliasFano<'c>> {
    let meta = c.u64s(&format!("{name}.m"))?;
    if meta.len() != 3 {
        return Err(CoreError::InvalidParts(format!(
            "`{name}.m` must hold 3 values"
        )));
    }
    let low_bits =
        u32::try_from(meta[1]).map_err(|_| CoreError::InvalidParts("low_bits".into()))?;
    let high = RankSelect::from_parts(
        Cow::Borrowed(c.u64s(&format!("{name}.hw"))?),
        Cow::Borrowed(c.u64s(&format!("{name}.hs"))?),
        meta[2] as usize,
    )?;
    EliasFano::from_parts(
        meta[0] as usize,
        low_bits,
        Cow::Borrowed(c.u64s(&format!("{name}.lo"))?),
        high,
    )
}

/// Adds all sections of a hypergraph under `prefix`.
pub fn write_hypergraph<'a>(
    w: &mut ContainerWriter<'a>,
    prefix: &str,
    g: &'a Hypergraph<'_>,
) -> Result<()> {
    let p = g.parts();
    w.add(&format!("{prefix}.node_prio"), Payload::U32(&p.node_prio))?;
    write_ef(w, &format!("{prefix}.node_ptr"), &p.node_ptr)?;
    w.add(&format!("{prefix}.inc_edge"), Payload::U32(&p.inc_edge))?;
    w.add(&format!("{prefix}.inc_role"), Payload::U8(&p.inc_role))?;
    write_ef(w, &format!("{prefix}.run_ptr"), &p.run_ptr)?;
    w.add(&format!("{prefix}.run_rel"), Payload::U16(&p.run_rel))?;
    w.add(&format!("{prefix}.run_start"), Payload::U32(&p.run_start))?;
    write_ef(w, &format!("{prefix}.edge_ptr"), &p.edge_ptr)?;
    w.add(&format!("{prefix}.edge_node"), Payload::U32(&p.edge_node))?;
    w.add(&format!("{prefix}.edge_role"), Payload::U8(&p.edge_role))?;
    w.add(&format!("{prefix}.edge_rel"), Payload::U16(&p.edge_rel))?;
    Ok(())
}

/// Reads a hypergraph written by [`write_hypergraph`], borrowing the mmap.
pub fn read_hypergraph<'c>(c: &'c Container, prefix: &str) -> Result<Hypergraph<'c>> {
    let parts = HypergraphParts {
        node_prio: Cow::Borrowed(c.u32s(&format!("{prefix}.node_prio"))?),
        node_ptr: read_ef(c, &format!("{prefix}.node_ptr"))?,
        inc_edge: Cow::Borrowed(c.u32s(&format!("{prefix}.inc_edge"))?),
        inc_role: Cow::Borrowed(c.bytes(&format!("{prefix}.inc_role"))?),
        run_ptr: read_ef(c, &format!("{prefix}.run_ptr"))?,
        run_rel: Cow::Borrowed(c.u16s(&format!("{prefix}.run_rel"))?),
        run_start: Cow::Borrowed(c.u32s(&format!("{prefix}.run_start"))?),
        edge_ptr: read_ef(c, &format!("{prefix}.edge_ptr"))?,
        edge_node: Cow::Borrowed(c.u32s(&format!("{prefix}.edge_node"))?),
        edge_role: Cow::Borrowed(c.bytes(&format!("{prefix}.edge_role"))?),
        edge_rel: Cow::Borrowed(c.u16s(&format!("{prefix}.edge_rel"))?),
    };
    Hypergraph::from_parts(parts)
}
