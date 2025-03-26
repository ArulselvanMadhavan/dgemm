use std::fs::File;

use protobuf::{CodedOutputStream, Message};
include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));
use perfetto::*;
use uuid::Uuid;

fn mk_track_desc() -> (TracePacket, TrackDescriptor) {
    let tp = TracePacket::new();
    let mut tdesc = TrackDescriptor::new();
    let p_uuid = Uuid::new_v4();
    let (_, p_lsb) = p_uuid.as_u64_pair();
    tdesc.set_uuid(p_lsb);
    (tp, tdesc)
}

// fn mk_process_desc(pid: i32, pname: String) -> MessageField<ProcessDescriptor> {
//     let mut pdesc = ProcessDescriptor::new();
//     pdesc.set_pid(pid);
//     pdesc.set_process_name(pname);
//     MessageField::some(pdesc)
// }

// fn mk_thread_desc(pid: i32, tid: i32, tname: String) -> MessageField<ThreadDescriptor> {
//     let mut tdesc = ThreadDescriptor::new();
//     tdesc.set_pid(pid);
//     tdesc.set_tid(tid);
//     tdesc.set_thread_name(tname);
//     MessageField::some(tdesc)
// }
const DIR: &str = "artifacts/trace";
pub fn clean_trace() {
    std::fs::remove_dir_all(DIR).unwrap();
    std::fs::create_dir_all(DIR).unwrap();
}
pub fn mk_trace_file(fname: &str) -> File {
    let fname = format!("{DIR}/{fname}", DIR = DIR, fname = fname);
    File::create(fname).unwrap()
}

pub fn write_trace(fname: &str, tpkts: Vec<TracePacket>) {
    let mut trace = Trace::new();
    trace.packet = tpkts;
    let mut file = mk_trace_file(fname);
    let mut cos = CodedOutputStream::new(&mut file);
    trace.write_to(&mut cos).unwrap();
    cos.flush().unwrap();
}

pub fn mk_time_slice(
    tid: u32,
    thread_uuid: u64,
    tname: &str,
    timestamps: [u64; 2],
) -> [TracePacket; 2] {
    let mut tpkts = [TracePacket::new(), TracePacket::new()];
    for (idx, tstamp) in timestamps.into_iter().enumerate() {
        tpkts[idx].set_timestamp(tstamp);
        tpkts[idx].set_trusted_packet_sequence_id(tid as u32);
        let mut tevt = TrackEvent::new();
        if idx == 0 {
            tevt.set_type(track_event::Type::TYPE_SLICE_BEGIN);
            tevt.set_name(tname.to_string());
        } else {
            tevt.set_type(track_event::Type::TYPE_SLICE_END);
            tevt.set_name(tname.to_string());
        }
        tevt.set_track_uuid(thread_uuid);
        tpkts[idx].set_track_event(tevt);
    }
    tpkts
}

pub fn slice_begin(tid: u32, thread_uuid: u64, tname: &str, timestamp: u64) -> TracePacket {
    let mut tpkt = TracePacket::new();
    tpkt.set_timestamp(timestamp);
    tpkt.set_trusted_packet_sequence_id(tid);
    let mut tevt = TrackEvent::new();
    tevt.set_type(track_event::Type::TYPE_SLICE_BEGIN);
    tevt.set_name(tname.to_string());
    tevt.set_track_uuid(thread_uuid);
    tpkt.set_track_event(tevt);
    tpkt
}

/// Caller is responsible for calling begin before calling end.
pub fn slice_end(tid: u32, thread_uuid: u64, timestamp: u64) -> TracePacket {
    let mut tpkt = TracePacket::new();
    tpkt.set_timestamp(timestamp);
    tpkt.set_trusted_packet_sequence_id(tid);
    let mut tevt = TrackEvent::new();
    tevt.set_type(track_event::Type::TYPE_SLICE_END);
    tevt.set_track_uuid(thread_uuid);
    tpkt.set_track_event(tevt);
    tpkt
}

pub fn get_trace_descriptors<const N: usize>(
    processes: Vec<(String, Vec<String>)>,
    desc_count: usize,
    thread_count: usize,
) -> Vec<[u64; N]> {
    // let mut pid = 0;
    let mut tpkts = Vec::with_capacity(desc_count);
    let mut tuuids = Vec::<[u64; N]>::with_capacity(thread_count);
    for p in processes.into_iter() {
        let (pname, threads) = p;
        let (mut tpkt, mut tdesc) = mk_track_desc();
        tdesc.set_static_name(pname);
        // tdesc.process = mk_process_desc(pid, pname);
        let root_uuid = tdesc.uuid();
        tpkt.set_track_descriptor(tdesc);
        tpkts.push(tpkt);
        // let mut tid = 0;
        for tname in threads.into_iter() {
            let (mut tpkt, mut tdesc) = mk_track_desc();
            tdesc.set_parent_uuid(root_uuid);
            let parent_uuid = tdesc.uuid();
            tdesc.set_static_name(tname.clone());
            // tdesc.thread = mk_thread_desc(pid, tid, tname);
            tpkt.set_track_descriptor(tdesc);
            tpkts.push(tpkt);
            let mut uuids = [0; N];
            for i in 0..N {
                let (mut tpkt, mut tdesc) = mk_track_desc();
                tdesc.set_static_name(format!("child{i}", i = i));
                tdesc.set_parent_uuid(parent_uuid);
                uuids[i] = tdesc.uuid();
                tpkt.set_track_descriptor(tdesc);
                tpkts.push(tpkt);
            }
            tuuids.push(uuids);
            // tuuids.push(thread_uuid);
            // Build Track events
            // let tpkt_pair = mk_time_slice(tid as u32, thread_uuid, "SAMPLE_EVT", [200, 250]);
            // tpkts.extend_from_slice(&tpkt_pair);
            // tid += 1;
        }
        // pid += 1;
    }
    write_trace("header_0_.perfetto", tpkts);
    tuuids
}
