use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Write, stdout};
use std::thread::sleep;
use std::time::Duration;

// === FIELD & POPULATION ===
const FIELD_SZ: usize = 20;
const POP: usize = 400;
const ELITE: usize = 80;
const MIN_N: usize = 60;
const MAX_NEURONS: usize = 1200;
const MAX_SYNAPSES: usize = 8000;
const MAX_STEPS: usize = 2000;
// === NEURO & HEBBIAN ===
const CH_DECAY: f64 = 0.65;
const FTG_REC: f64 = 0.12;
const FTG_FIRE: f64 = 0.22;
const HEB_WIN: usize = 8;
const PUN_WIN: usize = 12;
const STAG_LIM: u32 = 20;
// === REWARDS 
const FOOD_REWARD: f64 = 10000.0;
const REWARD_STRENGTH: f64 = 1.5;
// === EVOLUTION 
const HIGH_MUTATION_RATE: f64 = 0.25;
const LOW_MUTATION_RATE: f64 = 0.025;
const BREAKTHROUGH_MUTATION: f64 = 0.15;
const STABLE_GENS_REQUIRED: u32 = 30;
const RECORD_BREAK_APPLES: usize = 3;
const RECORD_FALL_TOLERANCE: usize = 10;
const DIVERSITY_POOL_SIZE: usize = 30;
const CHAMP_BANK: usize = 20;
const MODULE_CROSSOVER_RATE: f64 = 0.6;
const HORMONE_DECAY: f64 = 0.9;
const MAX_MODULES: u8 = 40;
const GENOME_CHECKSUM_LEN: usize = 4;
const ATTENTION_DIM: usize = 4;

// === ENCODING ===
fn u2b26(mut n: usize) -> String {
    if n == 0 { return "AA".to_string(); }
    let mut s = String::new();
    while n > 0 { s.push((b'A' + (n % 26) as u8) as char); n /= 26; }
    while s.len() < 2 { s.push('A'); }
    if s.len() > 2 { s = s[..2].to_string(); }
    s.chars().rev().collect()
}
fn b262u(s: &str) -> usize { s.chars().fold(0, |a, c| a * 26 + (c as u8 - b'A') as usize) }
fn f2bs(v: f64, lo: f64, hi: f64) -> String {
    let norm = ((v.clamp(lo, hi) - lo) / (hi - lo) * 675.0) as usize;
    format!("{}{}", if v >= 0.0 { 'P' } else { 'N' }, u2b26(norm.min(675)))
}
fn b2f(s: &str, lo: f64, hi: f64) -> f64 {
    if s.len() < 3 { return lo; }
    let sign: f64 = if s.chars().next() == Some('N') { -1.0 } else { 1.0 };
    let v = b262u(&s[1..3]) as f64 / 675.0;
    lo + v * (hi - lo) * sign.abs()
}


enum NType {
    Excitatory, Inhibitory, Effector,
    LoopDet, FoodSeek, WallAvoid, Randomizer,
    Memory, Speed, Risk, Novelty,
    HormoneSrc, HormoneSink, Oscillator, Gate,
    Query, Key, Value, AttentionOut, PlasticityMod,
}
use NType::*;

impl NType {
    fn ch(self) -> char {
        match self {
            Excitatory => 'E', Inhibitory => 'I', Effector => 'X',
            LoopDet => 'L', FoodSeek => 'F', WallAvoid => 'W', Randomizer => 'R',
            Memory => 'M', Speed => 'S', Risk => 'K', Novelty => 'V',
            HormoneSrc => 'H', HormoneSink => 'Z', Oscillator => 'O', Gate => 'G',
            Query => 'Q', Key => 'Y', Value => 'U', AttentionOut => 'A', PlasticityMod => 'P',
        }
    }
    fn from(c: char) -> Self {
        match c {
            'E' => Excitatory, 'I' => Inhibitory, 'X' => Effector,
            'L' => LoopDet, 'F' => FoodSeek, 'W' => WallAvoid, 'R' => Randomizer,
            'M' => Memory, 'S' => Speed, 'K' => Risk, 'V' => Novelty,
            'H' => HormoneSrc, 'Z' => HormoneSink, 'O' => Oscillator, 'G' => Gate,
            'Q' => Query, 'Y' => Key, 'U' => Value, 'A' => AttentionOut, 'P' => PlasticityMod,
            _ => Excitatory,
        }
    }
}


struct Neuron {
    id: u32, typ: NType, thr: f64, chg: f64, ftg: f64, fired: bool,
    hist: VecDeque<bool>,
    total: u64, food_ok: u64, w_death: u64, s_death: u64, l_death: u64, self_death: u64,
    age: u64, born: u64, mut_cnt: u32, last_tick: u64,
    util: f64, spec: usize, module: u8,
    hormone_level: f64, oscillator_phase: f64, gate_open: bool,
    q_vec: [f64; ATTENTION_DIM], k_vec: [f64; ATTENTION_DIM], v_vec: [f64; ATTENTION_DIM],
}
impl Neuron {
    fn new(id: u32, t: NType, thr: f64) -> Self {
        Self { id, typ: t, thr: thr.clamp(0.01,1.0), chg:0., ftg:0., fired:false,
            hist: VecDeque::from(vec![false;50]), total:0, food_ok:0, w_death:0, s_death:0, l_death:0, self_death:0,
            age:0, born:0, mut_cnt:0, last_tick:0, util:0., spec:0, module:0,
            hormone_level:0., oscillator_phase:0., gate_open:true,
            q_vec: [0.; ATTENTION_DIM], k_vec: [0.; ATTENTION_DIM], v_vec: [0.; ATTENTION_DIM] }
    }
    fn add_chg(&mut self, v: f64) { self.chg = (self.chg + v).clamp(-5.,5.); }
    fn should_fire(&self) -> bool {
        if matches!(self.typ, Effector) { return false; }
        if self.typ == Gate && !self.gate_open { return false; }
        self.chg >= (self.thr + self.ftg).min(0.99)
    }
    fn fire(&mut self) -> f64 {
        self.fired = true; self.total += 1; self.ftg = (self.ftg + FTG_FIRE).min(1.);
        let s = match self.typ { Inhibitory => -1., _ => 1. };
        self.chg = 0.;
        if self.typ == HormoneSrc { self.hormone_level = 1.; }
        else if self.typ == HormoneSink { self.hormone_level = 0.; }
        else if self.typ == Oscillator {
            self.oscillator_phase = (self.oscillator_phase + 0.1) % 1.0;
            self.hormone_level = if self.oscillator_phase > 0.5 { 1. } else { 0. };
        }
        s
    }
    fn end_tick(&mut self) {
        self.chg *= CH_DECAY; if self.chg.abs() < 1e-3 { self.chg = 0.; }
        self.ftg = (self.ftg - FTG_REC).max(0.);
        self.hormone_level *= HORMONE_DECAY;
        self.hist.pop_back(); self.hist.push_front(self.fired); self.fired = false;
    }
    fn active_in(&self, w: usize) -> bool { self.hist.iter().take(w).any(|&f| f) }
    fn compute_util(&mut self) {
        let g = self.food_ok as f64;
        let b = (self.w_death + self.s_death + self.l_death + self.self_death) as f64;
        let t = g + b;
        self.util = if t < 3. { 0. } else { (g - b * 0.5) / t };
    }
    fn reset_state(&mut self) {
        self.chg=0.; self.ftg=0.; self.fired=false;
        self.hist=VecDeque::from(vec![false;50]); self.last_tick=0;
        self.hormone_level=0.; self.oscillator_phase=0.;
    }
}


struct Synapse {
    from: u32, to: u32, str: f64, delay: u8, pend: VecDeque<f64>,
    used: u32, total_u: u64, food_ok: u64, w_death: u64, s_death: u64, l_death: u64, self_death: u64,
    rec: bool, modu: bool, mut_cnt: u32, util: f64, spec: usize, module: u8,
    hormone_sensitive: bool, attention_weight: f64,
}
impl Synapse {
    fn new(from: u32, to: u32, str: f64, dly: u8, rec: bool, modu: bool) -> Self {
        let mut p = VecDeque::new(); for _ in 0..=dly { p.push_back(0.); }
        Self { from, to, str: str.clamp(-1.,1.), delay: dly, pend: p,
            used:0, total_u:0, food_ok:0, w_death:0, s_death:0, l_death:0, self_death:0,
            rec, modu, mut_cnt:0, util:0., spec:0, module:0, hormone_sensitive:false, attention_weight:1.0 }
    }
    fn send(&mut self, v: f64, hormone_bonus: f64) {
        let bonus = if self.hormone_sensitive { 1. + hormone_bonus } else { 1. };
        self.pend[0] += v * self.str * bonus * self.attention_weight;
        self.used += 1; self.total_u += 1;
    }
    fn forward(&mut self) -> f64 { let o = self.pend.pop_back().unwrap_or(0.); self.pend.push_front(0.); o }
    fn strengthen(&mut self, a: f64) { if self.str >= 0. { self.str = (self.str + a).min(1.); } else { self.str = (self.str - a).max(-1.); } }
    fn weaken(&mut self, a: f64) { self.str *= 1. - a; }
    fn dead(&self) -> bool { self.str.abs() < 0.001 && self.total_u > 1000 }
    fn compute_util(&mut self) {
        let g = self.food_ok as f64;
        let b = (self.w_death + self.s_death + self.l_death + self.self_death) as f64;
        let t = g + b;
        self.util = if t < 3. { 0. } else { (g - b * 0.5) / t };
    }
    fn reset_state(&mut self) {
        self.pend = VecDeque::new(); for _ in 0..=self.delay { self.pend.push_back(0.); } self.used=0;
    }
}



impl Sensor { fn new(id: u32, t: SenT, d: u8) -> Self { Self { id, typ: t, dir: d, val: 0. } } }


struct Brain {
    neurons: HashMap<u32, Neuron>, syns: Vec<Synapse>, sens: Vec<Sensor>,
    eff: [u32;4], next_id: u32, ticks: u64, gen: u64,
    prev: HashMap<u32,f64>, specials: HashMap<String,u32>,
    stag_cnt: u32, novel: f64, spec_id: usize, ex_risk: f64,
    next_module: u8, global_hormone: f64, attention_context: Vec<f64>,
    plateau_counter: u32,
    last_best_fit: f64,
    visited_positions: HashMap<(i32,i32), u32>,
    path_memory: VecDeque<(i32,i32)>,
}
impl Brain {
    fn new() -> Self { 
        Self { 
            neurons: HashMap::new(), syns: vec![], sens: vec![], eff:[0;4], next_id:0, ticks:0, gen:0, 
            prev: HashMap::new(), specials: HashMap::new(), stag_cnt:0, novel:0., spec_id:0, ex_risk:0., 
            next_module:0, global_hormone:0., attention_context: vec![0.; ATTENTION_DIM], plateau_counter:0, 
            last_best_fit: f64::NEG_INFINITY,
            visited_positions: HashMap::new(),
            path_memory: VecDeque::with_capacity(100),
        } 
    }
    fn alloc(&mut self) -> u32 { let i = self.next_id; self.next_id += 1; i }
    fn alloc_module(&mut self) -> u8 { let m = self.next_module; self.next_module = self.next_module.wrapping_add(1); if self.next_module >= MAX_MODULES { self.next_module = 0; } m }
    
    fn ensure_minimum_circuits(&mut self) {
        if self.sens.is_empty() {
            for i in 0..24 { 
                let d = (i/3) as u8; 
                let t = match i%3 {0 => SenT::Wall,1 => SenT::Food,_ => SenT::Body}; 
                self.sens.push(Sensor::new(i as u32,t,d)); 
            }
        }
        
        if self.eff[0] == 0 {
            for i in 0..4 { 
                let id = self.alloc(); 
                self.eff[i] = id; 
                self.neurons.insert(id, Neuron::new(id, Effector, 0.)); 
            }
        }
        
        if self.neurons.len() < MIN_N {
            let current_len = self.neurons.len();
            for i in 0..(MIN_N - current_len) {
                let id = self.alloc(); 
                let m = self.alloc_module();
                let nt = if i % 3 == 0 { Memory } else if i % 3 == 1 { Randomizer } else { Excitatory };
                let mut n = Neuron::new(id, nt, 0.25);
                n.module = m;
                self.neurons.insert(id, n);
            }
        }
        
        let sens_data: Vec<(u32, u8, SenT)> = self.sens.iter()
            .map(|s| (s.id, s.dir, s.typ))
            .collect();
        let eff_ids = self.eff.to_vec();
        
        for (sid, dir, typ) in sens_data {
            let eidx = match dir { 0|4 => 0, 1|5 => 1, 2|6 => 2, _ => 3 };
            let eid = eff_ids[eidx];
            if !self.syns.iter().any(|sy| sy.from == sid && sy.to == eid) {
                let strength = match typ {
                    SenT::Wall => -0.8,
                    SenT::Food => 12.0,
                    SenT::Body => -1.5
                };
                let mut syn = Synapse::new(sid, eid, strength, 0, false, false);
                let m = self.alloc_module();
                syn.module = m;
                self.syns.push(syn);
            }
        }
    }

    fn ensure_specials(&mut self) {
        if self.sens.is_empty() { return; }
        let pairs = [
            ("loop", LoopDet), ("food", FoodSeek), ("wall", WallAvoid),
            ("random", Randomizer), ("memory", Memory), ("speed", Speed),
            ("risk", Risk), ("novelty", Novelty), ("hormone_src", HormoneSrc),
            ("hormone_sink", HormoneSink), ("osc", Oscillator),
            ("query", Query), ("key", Key), ("value", Value),
            ("attn_out", AttentionOut), ("plasticity", PlasticityMod),
        ];
        for &(name, typ) in &pairs {
            if !self.specials.contains_key(name) {
                let id = self.alloc(); let m = self.alloc_module();
                let mut n = Neuron::new(id, typ, 0.3);
                n.module = m;
                if matches!(typ, Query | Key | Value) {
                    let mut rng = rand::thread_rng();
                    for i in 0..ATTENTION_DIM {
                        n.q_vec[i] = rng.gen_range(-1.0..1.0);
                        n.k_vec[i] = rng.gen_range(-1.0..1.0);
                        n.v_vec[i] = rng.gen_range(-1.0..1.0);
                    }
                }
                self.neurons.insert(id, n);
                self.specials.insert(name.to_string(), id);
                match name {
                    "loop" => { for &e in &self.eff { let mut s = Synapse::new(id,e,-0.5,0,false,false); s.module=m; self.syns.push(s); } }
                    "food" => {
                        for s in 0..self.sens.len() { if self.sens[s].typ == SenT::Food { let mut sy=Synapse::new(s as u32,id,4.0,0,false,false); sy.module=m; self.syns.push(sy); } }
                        for &e in &self.eff { let mut sy=Synapse::new(id,e,2.5,0,false,false); sy.module=m; self.syns.push(sy); }
                    }
                    "wall" => {
                        for s in 0..self.sens.len() { if self.sens[s].typ == SenT::Wall { let mut sy=Synapse::new(s as u32,id,2.5,0,false,false); sy.module=m; self.syns.push(sy); } }
                        for &e in &self.eff { let mut sy=Synapse::new(id,e,-1.8,0,false,false); sy.module=m; self.syns.push(sy); }
                    }
                    "random" => { for &e in &self.eff { let mut sy=Synapse::new(id,e,1.5,0,false,false); sy.module=m; self.syns.push(sy); } }
                    "memory" => { 
                        for s in 0..self.sens.len() { let mut sy=Synapse::new(s as u32,id,1.5,0,true,false); sy.module=m; self.syns.push(sy); }
                        for &e in &self.eff { let mut sy=Synapse::new(id,e,1.2,0,true,false); sy.module=m; self.syns.push(sy); }
                    }
                    "hormone_src" => { for s in 0..self.sens.len() { let mut sy=Synapse::new(s as u32,id,0.8,0,false,false); sy.hormone_sensitive=true; sy.module=m; self.syns.push(sy); } }
                    "hormone_sink" => { for &e in &self.eff { let mut sy=Synapse::new(id,e,0.8,0,false,false); sy.hormone_sensitive=true; sy.module=m; self.syns.push(sy); } }
                    "osc" => { for &e in &self.eff { let mut sy=Synapse::new(id,e,1.0,0,false,false); sy.module=m; self.syns.push(sy); } }
                    "query" | "key" | "value" => {
                        for s in 0..self.sens.len() { let mut sy=Synapse::new(s as u32,id,1.0,0,false,false); sy.module=m; self.syns.push(sy); }
                        for n in self.neurons.values() { if n.typ == AttentionOut { let mut sy=Synapse::new(id, n.id, 1.2, 0, false, false); sy.module=m; self.syns.push(sy); } }
                    }
                    "attn_out" => { for &e in &self.eff { let mut sy=Synapse::new(id,e,1.5,0,false,false); sy.module=m; self.syns.push(sy); } }
                    "plasticity" => { for s in &mut self.syns { s.hormone_sensitive = true; } }
                    _ => { for &e in &self.eff { let mut sy=Synapse::new(id,e,1.0,0,false,false); sy.module=m; self.syns.push(sy); } }
                }
            }
        }
    }

    fn from_genome(gen: &str) -> Self {
        let mut b = Brain::new();
        b.ensure_minimum_circuits();
        
        let payload = if gen.len() > 4 { &gen[..gen.len()-4] } else { gen };
        for part in payload.split('|').filter(|p| !p.is_empty()) {
            let cmd = part.chars().next().unwrap();
            match cmd {
                'N' => {
                    if part.len() < 14 { continue; }
                    let id = b262u(&part[1..3]) as u32;
                    let tc = part.chars().nth(3).unwrap();
                    let th = b2f(&part[4..7], 0.01, 1.0);
                    let nt = NType::from(tc);
                    let module = part.chars().nth(13).map(|c| c as u8 - b'A').unwrap_or(b.alloc_module());
                    let mut n = Neuron::new(id, nt, th); n.module = module;
                    if part.len() >= 14 { n.total = b262u(&part[7..9]) as u64; n.food_ok = b262u(&part[9..11]) as u64; n.w_death = b262u(&part[11..13]) as u64; }
                    if matches!(nt, Query | Key | Value) && part.len() >= 14 + ATTENTION_DIM * 9 {
                        let mut idx = 14;
                        for i in 0..ATTENTION_DIM { if idx+3 <= part.len() { n.q_vec[i] = b2f(&part[idx..idx+3], -1.0, 1.0); idx += 3; } }
                        for i in 0..ATTENTION_DIM { if idx+3 <= part.len() { n.k_vec[i] = b2f(&part[idx..idx+3], -1.0, 1.0); idx += 3; } }
                        for i in 0..ATTENTION_DIM { if idx+3 <= part.len() { n.v_vec[i] = b2f(&part[idx..idx+3], -1.0, 1.0); idx += 3; } }
                    }
                    b.neurons.insert(id, n); b.next_id = b.next_id.max(id+1);
                    match tc { 'U' => b.eff[0]=id, 'D' => b.eff[1]=id, 'L' => b.eff[2]=id, 'R' => b.eff[3]=id, _ => {} }
                    if tc=='L' { b.specials.insert("loop".into(), id); } else if tc=='F' { b.specials.insert("food".into(), id); }
                    else if tc=='W' { b.specials.insert("wall".into(), id); } else if tc=='R' { b.specials.insert("random".into(), id); }
                    else if tc=='M' { b.specials.insert("memory".into(), id); } else if tc=='S' { b.specials.insert("speed".into(), id); }
                    else if tc=='K' { b.specials.insert("risk".into(), id); } else if tc=='V' { b.specials.insert("novelty".into(), id); }
                    else if tc=='H' { b.specials.insert("hormone_src".into(), id); } else if tc=='Z' { b.specials.insert("hormone_sink".into(), id); }
                    else if tc=='O' { b.specials.insert("osc".into(), id); }
                    else if tc=='Q' { b.specials.insert("query".into(), id); } else if tc=='Y' { b.specials.insert("key".into(), id); }
                    else if tc=='U' { b.specials.insert("value".into(), id); } else if tc=='A' { b.specials.insert("attn_out".into(), id); }
                    else if tc=='P' { b.specials.insert("plasticity".into(), id); }
                },
                'S' => {
                    if part.len() < 18 { continue; }
                    let from = b262u(&part[1..3]) as u32; let to = b262u(&part[3..5]) as u32;
                    let str = b2f(&part[5..8], -1.0, 1.0);
                    let dly = (part.chars().nth(8).unwrap() as u8 - b'A') as u8;
                    let rec = part.chars().nth(9).unwrap() == 'R';
                    let modu = part.chars().nth(10).unwrap() == 'M';
                    let module = part.chars().nth(17).map(|c| c as u8 - b'A').unwrap_or(b.alloc_module());
                    let mut s = Synapse::new(from, to, str, dly, rec, modu); s.module = module;
                    if part.len() >= 18 { s.total_u = b262u(&part[11..13]) as u64; s.food_ok = b262u(&part[13..15]) as u64; s.w_death = b262u(&part[15..17]) as u64; }
                    if part.len() >= 21 { s.attention_weight = b2f(&part[18..21], 0.1, 2.0); }
                    b.syns.push(s);
                },
                _ => {}
            }
        }
        b.ensure_specials();
        b.ensure_minimum_circuits();
        b
    }

    fn compute_attention(&mut self) {
        let mut queries = Vec::new(); let mut keys = Vec::new(); let mut values = Vec::new();
        for n in self.neurons.values() {
            if n.typ == Query && n.fired { queries.push(n.q_vec); }
            if n.typ == Key && n.fired { keys.push(n.k_vec); }
            if n.typ == Value && n.fired { values.push(n.v_vec); }
        }
        if queries.is_empty() || keys.is_empty() || values.is_empty() { return; }
        let mut total_weight = 0.0;
        let mut weighted_values = [0.; ATTENTION_DIM];
        for qvec in &queries {
            for kvec in &keys {
                let mut dot = 0.0;
                for i in 0..ATTENTION_DIM { dot += qvec[i] * kvec[i]; }
                let weight = (dot + 1.0).max(0.0);
                total_weight += weight;
                for vvec in &values { for i in 0..ATTENTION_DIM { weighted_values[i] += weight * vvec[i]; } }
            }
        }
        if total_weight > 0.0 {
            for i in 0..ATTENTION_DIM { weighted_values[i] /= total_weight; }
            self.attention_context = weighted_values.to_vec();
        }
        for s in &mut self.syns {
            if self.neurons.get(&s.from).map(|n| n.typ == AttentionOut).unwrap_or(false) {
                s.attention_weight = self.attention_context.iter().sum::<f64>().abs().clamp(0.1, 2.0);
            }
        }
    }

    fn is_safe_position(&self, pos: (i32,i32), body: &VecDeque<(i32,i32)>) -> bool {
        pos.0 >= 0 && pos.0 < FIELD_SZ as i32 && 
        pos.1 >= 0 && pos.1 < FIELD_SZ as i32 &&
        !body.contains(&pos)
    }

    fn tick(&mut self, sv: &[f64;24], body: &VecDeque<(i32,i32)>, len: usize, eaten: usize, turns: usize, looping: bool, head: (i32,i32), _food_pos: (i32,i32)) -> usize {
        self.ensure_specials();
        self.ensure_minimum_circuits();
        
        self.path_memory.push_back(head);
        if self.path_memory.len() > 50 { self.path_memory.pop_front(); }
        
        *self.visited_positions.entry(head).or_insert(0) += 1;
        
        for e in &self.eff {
            if let Some(n) = self.neurons.get_mut(e) {
                n.chg = 0.;
            }
        }
        
        let mut best_food_dir = 4;
        let mut best_food_val = 0.0;
        for dir_idx in 0..4 {
            let food_val = sv[dir_idx * 3 + 1];
            if food_val > best_food_val {
                best_food_val = food_val;
                best_food_dir = dir_idx;
            }
        }
        
        if best_food_val > 0.03 {
            let (dx, dy) = Dir::from(best_food_dir).dxy();
            let new_pos = (head.0 + dx, head.1 + dy);
            
            if self.is_safe_position(new_pos, body) {
                if let Some(n) = self.neurons.get_mut(&self.eff[best_food_dir]) {
                    n.chg = 999.0;
                }
                return best_food_dir;
            } else {
                for dir_idx in 0..4 {
                    let (dx, dy) = Dir::from(dir_idx).dxy();
                    let new_pos = (head.0 + dx, head.1 + dy);
                    if self.is_safe_position(new_pos, body) && new_pos != head {
                        if let Some(n) = self.neurons.get_mut(&self.eff[dir_idx]) {
                            n.chg = 500.0;
                        }
                        return dir_idx;
                    }
                }
            }
        }
        
        for (i,s) in self.sens.iter_mut().enumerate() { 
            s.val = sv[i].clamp(0.,1.); 
            if s.typ == SenT::Body && len > 5 {
                s.val *= 3.0;
            }
        }
        self.global_hormone = 0.;
        for n in self.neurons.values() {
            if n.typ == HormoneSrc && n.fired { self.global_hormone += n.hormone_level; }
            if n.typ == HormoneSink { self.global_hormone = 0.; }
        }
        
        for n in self.neurons.values_mut() {
            match n.typ {
                LoopDet => n.chg = if looping {2.} else {0.},
                FoodSeek => n.chg = if eaten < 3 { 3.0 } else { 0.0 },
                WallAvoid => n.chg = if len > 8 { 2.0 } else { 0.0 },
                Randomizer => n.chg = if turns < 2 { 1.5 } else if len > 20 { 2.0 } else { 0.0 },
                Novelty => {
                    let visits = *self.visited_positions.get(&head).unwrap_or(&0);
                    n.chg = if visits < 2 { 2.0 } else { 0.0 };
                }
                Oscillator => { n.oscillator_phase = (n.oscillator_phase + 0.1) % 1.0; n.chg = if n.oscillator_phase > 0.5 {1.} else {0.}; }
                Gate => n.gate_open = self.global_hormone > 0.4,
                _ => {}
            }
        }
        
        let mut inputs = vec![0.; self.neurons.len() + 200];
        for s in &mut self.syns { let o = s.forward(); if o!=0. { let idx = s.to as usize; if idx < inputs.len() { inputs[idx] += o; } } }
        for s in &self.sens { if s.val > 0.01 { for syn in &mut self.syns { if syn.from == s.id { syn.send(s.val, self.global_hormone); } } } }
        for syn in &mut self.syns { if syn.rec { if let Some(&p) = self.prev.get(&syn.from) { if p > 0.01 { syn.send(p, self.global_hormone); } } } }
        for (id,n) in &self.neurons { if n.chg > 0.5 && matches!(n.typ, LoopDet|FoodSeek|WallAvoid|Randomizer|Memory|Speed|Risk|Novelty|HormoneSrc|HormoneSink|Oscillator|Query|Key|Value|AttentionOut|PlasticityMod) { for syn in &mut self.syns { if syn.from == *id { syn.send(n.chg*0.8, self.global_hormone); } } } }
        for (id,n) in self.neurons.iter_mut() { let idx = *id as usize; if idx < inputs.len() && inputs[idx]!=0. { n.add_chg(inputs[idx]); } }
        let nids: Vec<u32> = self.neurons.keys().copied().collect();
        for &id in &nids { if let Some(n) = self.neurons.get_mut(&id) { if n.should_fire() { let s = n.fire(); if s!=0. { for syn in &mut self.syns { if syn.from == id { syn.send(s, self.global_hormone); } } } } } }
        self.compute_attention();
        
        let mut max = -1000.; let mut chosen = 0;
        for (i, &e) in self.eff.iter().enumerate() {
            if let Some(n) = self.neurons.get(&e) { if n.chg > max { max = n.chg; chosen = i; } }
        }
        if max <= 0.1 { 
            let safe_dirs: Vec<usize> = (0..4).filter(|&d| {
                let (dx, dy) = Dir::from(d).dxy();
                let new_pos = (head.0 + dx, head.1 + dy);
                self.is_safe_position(new_pos, body)
            }).collect();
            if !safe_dirs.is_empty() {
                chosen = *safe_dirs.choose(&mut rand::thread_rng()).unwrap();
            } else {
                chosen = rand::thread_rng().gen_range(0..4);
            }
        }
        
        self.prev.clear();
        for (&id,n) in &self.neurons { self.prev.insert(id, n.chg); }
        for n in self.neurons.values_mut() { n.end_tick(); }
        self.ticks += 1;
        chosen
    }

    fn reset_state(&mut self) { 
        for n in self.neurons.values_mut() { n.reset_state(); } 
        for s in &mut self.syns { s.reset_state(); } 
        self.prev.clear(); 
        self.ticks=0; 
        self.novel=0.; 
        self.global_hormone=0.; 
        self.attention_context = vec![0.; ATTENTION_DIM];
        self.visited_positions.clear();
        self.path_memory.clear();
    }
    
    fn reward(&mut self) {
        for n in self.neurons.values_mut() { 
            if n.active_in(HEB_WIN) { 
                n.food_ok += 1; 
                n.thr = (n.thr - 0.02).max(0.01);
                n.util += 0.1;
            } 
        }
        for s in &mut self.syns { 
            if s.used > 0 { 
                s.food_ok += 1; 
                s.strengthen(REWARD_STRENGTH); 
                if self.neurons.get(&s.to).map(|n| matches!(n.typ, Effector)).unwrap_or(false) { 
                    s.strengthen(REWARD_STRENGTH * 0.5); 
                } 
            } 
        }
    }
    
    fn punish(&mut self, reason: &str) {
        let pw = match reason { "wall_collision" => 2.0, "starvation" => 1.5, "looping" => 2.0, "self_collision" => 2.5, _ => 1.0 };
        for n in self.neurons.values_mut() { 
            if n.active_in(PUN_WIN) { 
                match reason { 
                    "wall_collision" => n.w_death+=1, 
                    "starvation" => n.s_death+=1, 
                    "looping" => n.l_death+=1, 
                    "self_collision" => n.self_death+=1, 
                    _ => {} 
                } 
                n.util -= 0.2;
            } 
        }
        for s in &mut self.syns { 
            if s.used > 0 { 
                match reason { 
                    "wall_collision" => s.w_death+=1, 
                    "starvation" => s.s_death+=1, 
                    "looping" => s.l_death+=1, 
                    "self_collision" => s.self_death+=1, 
                    _ => {} 
                } 
                s.weaken(0.05 * pw); 
            } 
        }
    }

    fn deep_mutate(&mut self, _reason: &str, gen: u64, _stats: &SnakeStats, stagn: bool, _eaten: usize, plateau: u32, mut_rate: f64) {
        self.ensure_specials(); 
        self.ensure_minimum_circuits();
        let mut rng = rand::thread_rng();
        
        let mut ch = mut_rate;
        if stagn { ch *= 1.5; }
        if plateau > 15 { ch *= 1.3; }
        
        let new_neurons = rng.gen_range(1..=3);
        for _ in 0..new_neurons {
            if self.neurons.len() >= MAX_NEURONS { break; }
            let id = self.alloc(); let m = self.alloc_module();
            let nt = match rng.gen_range(0..10) {
                0 => Randomizer,
                1 => FoodSeek,
                2 => Novelty,
                3 => Memory,
                4 => Risk,
                5 => LoopDet,
                _ => Excitatory
            };
            let mut n = Neuron::new(id, nt, rng.gen_range(0.1..0.4)); 
            n.module = m; 
            self.neurons.insert(id, n);
            
            let neuron_ids: Vec<u32> = self.neurons.keys().copied().collect();
            if neuron_ids.len() > 1 {
                let from = neuron_ids[rng.gen_range(0..neuron_ids.len())];
                let to = id;
                self.syns.push(Synapse::new(from, to, rng.gen_range(-0.5..0.5), 0, rng.gen_bool(0.1), false));
                
                let from = id;
                let to = neuron_ids[rng.gen_range(0..neuron_ids.len())];
                if from != to {
                    self.syns.push(Synapse::new(from, to, rng.gen_range(-0.5..0.5), 0, rng.gen_bool(0.1), false));
                }
            }
        }
        
        for s in &mut self.syns {
            if rng.gen_bool(ch * 0.3) {
                s.str = (s.str + rng.gen_range(-0.1..0.1)).clamp(-1.0, 1.0);
                s.mut_cnt += 1;
            }
            if rng.gen_bool(ch * 0.05) {
                s.rec = !s.rec;
            }
        }
        
        if self.syns.len() > MAX_SYNAPSES {
            let excess = self.syns.len() - MAX_SYNAPSES;
            let to_remove: Vec<usize> = self.syns.iter()
                .enumerate()
                .filter(|(_, s)| s.str.abs() < 0.01 && s.total_u > 1000)
                .take(excess.min(10))
                .map(|(i, _)| i)
                .collect();
            for i in to_remove.iter().rev() {
                if *i < self.syns.len() {
                    self.syns.remove(*i);
                }
            }
        }
        
        for n in self.neurons.values_mut() { n.compute_util(); n.age += 1; }
        for s in &mut self.syns { s.compute_util(); }
        
        self.stag_cnt = if stagn { 0 } else { self.stag_cnt + 1 }; 
        self.gen = gen;
    }

    fn to_genome(&self) -> String {
        let mut p = Vec::new();
        for (&id, n) in &self.neurons {
            let tc = match n.typ { Excitatory => 'E', Inhibitory => 'I', LoopDet => 'L', FoodSeek => 'F', WallAvoid => 'W', Randomizer => 'R', Memory => 'M', Speed => 'S', Risk => 'K', Novelty => 'V', HormoneSrc => 'H', HormoneSink => 'Z', Oscillator => 'O', Gate => 'G', Query => 'Q', Key => 'Y', Value => 'U', AttentionOut => 'A', PlasticityMod => 'P', Effector => if id==self.eff[0] {'U'} else if id==self.eff[1] {'D'} else if id==self.eff[2] {'L'} else {'R'} };
            let mut entry = format!("N{}{}{}{}{}{}", u2b26(id as usize), tc, f2bs(n.thr,0.01,1.0), u2b26((n.total as usize).min(675)), u2b26((n.food_ok as usize).min(675)), u2b26((n.w_death as usize).min(675)));
            entry.push((b'A' + n.module) as char);
            if matches!(n.typ, Query | Key | Value) {
                for i in 0..ATTENTION_DIM { entry.push_str(&f2bs(n.q_vec[i], -1.0, 1.0)); }
                for i in 0..ATTENTION_DIM { entry.push_str(&f2bs(n.k_vec[i], -1.0, 1.0)); }
                for i in 0..ATTENTION_DIM { entry.push_str(&f2bs(n.v_vec[i], -1.0, 1.0)); }
            }
            p.push(entry);
        }
        for s in &self.syns {
            if s.str.abs() > 1e-3 {
                let mut entry = format!("S{}{}{}{}{}{}{}{}{}", u2b26(s.from as usize), u2b26(s.to as usize), f2bs(s.str,-1.,1.), (b'A'+s.delay) as char, if s.rec{'R'}else{'F'}, if s.modu{'M'}else{'S'}, u2b26((s.total_u as usize).min(675)), u2b26((s.food_ok as usize).min(675)), u2b26((s.w_death as usize).min(675)));
                entry.push((b'A' + s.module) as char);
                entry.push_str(&f2bs(s.attention_weight, 0.1, 2.0));
                p.push(entry);
            }
        }
        let payload = p.join("|");
        let checksum = u2b26(payload.len() % 456976);
        format!("{}{}", payload, checksum)
    }

    fn swap_modules(&mut self, other: &Brain) {
        let mut rng = rand::thread_rng();
        let ms: Vec<u8> = self.neurons.values().map(|n| n.module).chain(self.syns.iter().map(|s| s.module)).collect();
        let mo: Vec<u8> = other.neurons.values().map(|n| n.module).chain(other.syns.iter().map(|s| s.module)).collect();
        if ms.is_empty() || mo.is_empty() { return; }
        if rng.gen_bool(MODULE_CROSSOVER_RATE) {
            let m_self = ms[rng.gen_range(0..ms.len())];
            let m_other = mo[rng.gen_range(0..mo.len())];
            for n in self.neurons.values_mut() { if n.module == m_self { n.module = m_other; } }
            for s in &mut self.syns { if s.module == m_self { s.module = m_other; } }
        }
    }
}


impl Dir {
    fn opp(self) -> Self { match self { Dir::Up => Dir::Down, Dir::Down => Dir::Up, Dir::Left => Dir::Right, Dir::Right => Dir::Left } }
    fn dxy(self) -> (i32,i32) { match self { Dir::Up => (0,-1), Dir::Down => (0,1), Dir::Left => (-1,0), Dir::Right => (1,0) } }
    fn from(i: usize) -> Self { match i {0 => Dir::Up,1 => Dir::Down,2 => Dir::Left,_ => Dir::Right} }
}

struct Snake {
    body: VecDeque<(i32,i32)>, dir: Dir, food: (i32,i32),
    steps: usize, eaten: usize, turns: usize, alive: bool, reason: String,
    pos_hist: VecDeque<(i32,i32)>, loop_cnt: usize, stats: SnakeStats, loop_det: bool,
}
impl Snake {
    fn new() -> Self {
        let mut rng = rand::thread_rng(); let sx = rng.gen_range(5..15); let sy = rng.gen_range(5..15);
        let mut b = VecDeque::new(); b.push_back((sx,sy)); b.push_back((sx-1,sy)); b.push_back((sx-2,sy));
        let mut s = Self { body:b, dir:Dir::Right, food:(0,0), steps:0, eaten:0, turns:0, alive:true, reason:String::new(), pos_hist:VecDeque::new(), loop_cnt:0, stats:SnakeStats::default(), loop_det:false };
        s.spawn_food(); s
    }
    fn spawn_food(&mut self) { 
        let mut rng = rand::thread_rng(); 
        loop { 
            let x = rng.gen_range(0..FIELD_SZ as i32); 
            let y = rng.gen_range(0..FIELD_SZ as i32); 
            if !self.body.contains(&(x,y)) { 
                self.food = (x,y); 
                break; 
            } 
        } 
    }
    
    fn sense(&self) -> [f64;24] {
        let head = *self.body.front().unwrap();
        let off = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)];
        let mut v = [0.;24];
        let fdx = self.food.0 - head.0;
        let fdy = self.food.1 - head.1;
        let fdist = (fdx.abs() + fdy.abs()) as f64;
        
        for (di, (dx, dy)) in off.iter().enumerate() {
            let mut w = 0;
            let mut b = 0;
            let mut x = head.0;
            let mut y = head.1;
            
            for step in 1..FIELD_SZ as i32 {
                x += dx;
                y += dy;
                if x < 0 || x >= FIELD_SZ as i32 || y < 0 || y >= FIELD_SZ as i32 {
                    w = step;
                    break;
                }
                if self.body.contains(&(x, y)) && b == 0 {
                    b = step;
                }
            }
            v[di*3] = w as f64 / FIELD_SZ as f64;
            v[di*3+2] = b as f64 / FIELD_SZ as f64;
            
            let dir_aligned = (fdx.signum() == dx.signum() || *dx == 0) 
                           && (fdy.signum() == dy.signum() || *dy == 0);
            if dir_aligned && fdist > 0.0 {
                v[di*3+1] = 1.0 / (1.0 + fdist);
            } else {
                v[di*3+1] = 0.0;
            }
        }
        v
    }

    fn update(&mut self, dec: usize) {
        if !self.alive { return; }
        let nd = Dir::from(dec);
        if nd != self.dir.opp() { if nd != self.dir { self.turns+=1; } self.dir = nd; }
        let (dx,dy) = self.dir.dxy(); let h = *self.body.front().unwrap(); let nh = (h.0+dx, h.1+dy);
        if nh.0<0||nh.0>=FIELD_SZ as i32||nh.1<0||nh.1>=FIELD_SZ as i32 { self.alive=false; self.reason="wall_collision".into(); return; }
        if self.body.contains(&nh) { self.alive=false; self.reason="self_collision".into(); return; }
        self.body.push_front(nh);
        if nh == self.food { self.eaten+=1; self.steps=0; self.spawn_food(); } else { self.body.pop_back(); self.steps+=1; }
        self.pos_hist.push_back(nh); if self.pos_hist.len() > 20 { self.pos_hist.pop_front(); }
        self.stats.repeated = 0; self.loop_det = false;
        if self.pos_hist.len() >= 8 { let first = self.pos_hist[0]; self.stats.repeated = self.pos_hist.iter().filter(|&&p| p==first).count(); if self.stats.repeated >= 4 { self.loop_cnt += 1; self.loop_det = true; } }
        self.stats.no_food = self.steps;
        if self.steps > MAX_STEPS { self.alive=false; self.reason="starvation".into(); }
        if self.loop_cnt > 15 { self.alive=false; self.reason="looping".into(); }
    }
}


struct Genome { seq: String, fit: f64, reason: String, stats: SnakeStats, brain: Option<Brain>, novel: f64 }
impl Genome {
    fn random() -> Self {
        let mut b = Brain::new(); 
        b.ensure_specials();
        b.ensure_minimum_circuits();
        Genome{seq:b.to_genome(),fit:0.,reason:String::new(),stats:SnakeStats::default(),brain:Some(b),novel:0.}
    }
    fn crossover(p1: &Genome, p2: &Genome, reason: &str, gen: u64, stats: &SnakeStats, stagn: bool, eaten: usize, plateau: u32, mut_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut b1 = p1.brain.clone().unwrap_or_else(Brain::new);
        let b2 = p2.brain.clone().unwrap_or_else(Brain::new);
        
        let mut child_syns = Vec::new();
        for s in b1.syns.iter().chain(b2.syns.iter()) { 
            if rng.gen_bool(0.5) { 
                child_syns.push(s.clone()); 
            } 
        }
        child_syns.sort_by_key(|s| (s.from, s.to)); 
        child_syns.dedup_by_key(|s| (s.from, s.to));
        b1.syns = child_syns;
        
        for (&id, n) in &b2.neurons { 
            if !b1.neurons.contains_key(&id) { 
                b1.neurons.insert(id, n.clone()); 
            } 
        }
        
        b1.swap_modules(&b2); 
        b1.gen = gen; 
        b1.deep_mutate(reason, gen, stats, stagn, eaten, plateau, mut_rate); 
        b1.ensure_minimum_circuits();
        
        Genome{seq:b1.to_genome(),fit:0.,reason:reason.to_string(),stats:stats.clone(),brain:Some(b1),novel:0.}
    }
}

fn evaluate(g: &mut Genome, gen: u64) -> (f64, usize) {
    let base_brain = g.brain.take().unwrap_or_else(|| Brain::from_genome(&g.seq));
    g.fit = 0.0;
    let mut deaths: Vec<String> = Vec::new();
    let mut all_stats = SnakeStats::default();
    let mut max_food = 0;

    for ep in 0..2 {
        let mut b = if ep == 0 { base_brain.clone() } else { Brain::from_genome(&g.seq) };
        b.gen = gen; b.reset_state();
        let mut sn = Snake::new(); 
        let mut pf = 0;
        
        while sn.alive {
            let head = *sn.body.front().unwrap();
            let sv = sn.sense(); 
            let dec = b.tick(&sv, &sn.body, sn.body.len(), sn.eaten, sn.turns, sn.loop_det, head, sn.food);
            sn.update(dec);
            
            if sn.eaten > pf { 
                b.reward();
                let bonus = (sn.eaten as f64).powf(2.0) * 2000.0;
                g.fit += bonus;
                pf = sn.eaten;
                max_food = max_food.max(sn.eaten);
            }
            
            g.fit -= 0.3;
        }
        
        g.fit += (sn.body.len() as f64) * 80.0;
        
        match sn.reason.as_str() {
            "wall_collision" => g.fit -= 200.0,
            "self_collision" => g.fit -= 300.0,
            "starvation" => g.fit -= 150.0,
            "looping" => g.fit -= 250.0,
            _ => {}
        }
        
        b.punish(&sn.reason);
        deaths.push(sn.reason.clone());
        all_stats.repeated += sn.stats.repeated; 
        all_stats.no_food += sn.stats.no_food;
    }

    if max_food >= 100 {
        g.fit += 500000.0;
    } else if max_food >= 90 {
        g.fit += 300000.0;
    } else if max_food >= 80 {
        g.fit += 150000.0;
    } else if max_food >= 70 {
        g.fit += 80000.0;
    } else if max_food >= 60 {
        g.fit += 40000.0;
    } else if max_food >= 50 {
        g.fit += 20000.0;
    } else if max_food >= 40 {
        g.fit += 10000.0;
    } else if max_food >= 30 {
        g.fit += 5000.0;
    } else if max_food >= 20 {
        g.fit += 1000.0;
    }

    let main_reason = deaths.iter().max_by_key(|r| deaths.iter().filter(|&x| x==*r).count()).unwrap_or(&String::new()).clone();
    g.reason = main_reason; 
    g.stats = all_stats;
    
    if let Some(ref mut b) = g.brain {
        if (g.fit - b.last_best_fit).abs() < 5000.0 {
            b.plateau_counter += 1;
        } else {
            b.plateau_counter = 0;
            b.last_best_fit = g.fit;
        }
    }
    
    g.brain = Some(base_brain);
    (g.fit, max_food)
}

fn render(b: &Brain, sn: &Snake, gen: usize, best: f64, max_food: usize, mut_rate: f64) {
    print!("\x1B[2J\x1B[1;1H");
    println!("Gen:{} | Best:{:.0} | MaxFood:{} | N:{} | S:{} | Death:{} | Steps:{} | Mut:{:.3} | Plateau:{}", 
        gen, best, max_food, b.neurons.len(), b.syns.len(), sn.reason, sn.steps, mut_rate, b.plateau_counter);
    println!("{}", "═".repeat(FIELD_SZ+2));
    for y in 0..FIELD_SZ as i32 { 
        print!("║"); 
        for x in 0..FIELD_SZ as i32 { 
            if sn.body.front() == Some(&(x,y)) { print!("@ "); } 
            else if sn.body.contains(&(x,y)) { print!("o "); } 
            else if sn.food == (x,y) { print!("* "); } 
            else { print!("  "); } 
        } 
        println!("║"); 
    }
    println!("{}", "═".repeat(FIELD_SZ+2));
}

fn main() {
    println!("=== BIO BRAIN - GRADUAL EVOLUTION ===");
    println!("HIGH mutation rate: {:.3}", HIGH_MUTATION_RATE);
    println!("LOW mutation rate:  {:.3} (10x lower)", LOW_MUTATION_RATE);
    println!("Stable record required: {} generations", STABLE_GENS_REQUIRED);
    println!("1=New 2=Watch 3=Continue"); print!("> "); stdout().flush().unwrap();
    let mut s = String::new(); std::io::stdin().read_line(&mut s).unwrap();
    if s.trim() == "2" {
        if let Ok(data) = std::fs::read_to_string("best_bio.txt") { 
            let mut b = Brain::from_genome(&data); 
            let mut sn = Snake::new(); 
            loop { 
                let head = *sn.body.front().unwrap();
                let sv = sn.sense(); 
                let dec = b.tick(&sv, &sn.body, sn.body.len(), sn.eaten, sn.turns, sn.loop_det, head, sn.food); 
                sn.update(dec); 
                render(&b, &sn, 0, 0., sn.eaten, LOW_MUTATION_RATE); 
                if !sn.alive { 
                    println!("Died: {} | Food: {}", sn.reason, sn.eaten);
                    sleep(Duration::from_secs(1));
                    sn = Snake::new(); 
                } 
                sleep(Duration::from_millis(20)); 
            } 
        }
        return;
    }
    let mut pop: Vec<Genome>; let mut gen: u64 = 0; let mut best_fit = f64::NEG_INFINITY; let mut stag_cnt = 0u32;
    let mut hall: Vec<Genome> = Vec::new();
    let mut best_food_ever = 0;
    
    let mut current_mut_rate = HIGH_MUTATION_RATE;
    let mut record_holding_gens = 0;
    let mut last_record_food = 0;
    let mut diversity_pool: VecDeque<Genome> = VecDeque::with_capacity(DIVERSITY_POOL_SIZE);
    let mut snowball_counter = 0;
    
    if s.trim() == "3" {
        if let Ok(data) = std::fs::read_to_string("best_bio.txt") {
            let base_b = Brain::from_genome(&data);
            let base = Genome{seq:data,fit:0.,reason:String::new(),stats:SnakeStats::default(),brain:Some(base_b),novel:0.};
            pop = Vec::new(); 
            for _ in 0..ELITE { pop.push(base.clone()); }
            while pop.len() < POP { 
                let mut b = base.brain.clone().unwrap(); 
                b.gen = gen; 
                b.deep_mutate("starvation", gen, &SnakeStats::default(), false, 0, 0, HIGH_MUTATION_RATE); 
                pop.push(Genome{seq:b.to_genome(),fit:0.,reason:String::new(),stats:SnakeStats::default(),brain:Some(b),novel:0.}); 
            }
            last_record_food = 85;
            best_food_ever = 85;
        } else { println!("No save. Starting new."); pop = (0..POP).map(|_| Genome::random()).collect(); }
    } else { pop = (0..POP).map(|_| Genome::random()).collect(); }
    
    loop {
        let mut max_food_in_gen = 0;
        for g in &mut pop { 
            let (fit, food) = evaluate(g, gen);
            g.fit = fit;
            max_food_in_gen = max_food_in_gen.max(food);
        }
        pop.sort_by(|a,b| b.fit.partial_cmp(&a.fit).unwrap_or(std::cmp::Ordering::Equal));
        
        if max_food_in_gen > best_food_ever {
            best_food_ever = max_food_in_gen;
            println!(" ⭐⭐⭐ NEW RECORD: {} apples! ⭐⭐⭐", best_food_ever);
        }
        
        
        if max_food_in_gen >= last_record_food + RECORD_BREAK_APPLES {
            snowball_counter += 1;
            last_record_food = max_food_in_gen;
            record_holding_gens = 0;
            current_mut_rate = LOW_MUTATION_RATE;
            
            diversity_pool.push_back(pop[0].clone());
            if diversity_pool.len() > DIVERSITY_POOL_SIZE {
                diversity_pool.pop_front();
            }
            
            println!(" ❄️ SNOWBALL #{}: {} apples! Switching to LOW mutation ({:.3})", 
                     snowball_counter, last_record_food, current_mut_rate);
            
        } else if max_food_in_gen >= last_record_food - RECORD_FALL_TOLERANCE && record_holding_gens < STABLE_GENS_REQUIRED {
            record_holding_gens += 1;
            if record_holding_gens % 5 == 0 {
                println!(" 📊 Stable record for {} generations (mut: {:.3}, best: {} apples)", 
                         record_holding_gens, current_mut_rate, max_food_in_gen);
            }
            
            if max_food_in_gen < last_record_food - 8 {
                println!(" ⚠️ Performance drop! Boosting mutation...");
                current_mut_rate = HIGH_MUTATION_RATE;
            }
            
        } else if record_holding_gens >= STABLE_GENS_REQUIRED {
            
            current_mut_rate = HIGH_MUTATION_RATE;
            record_holding_gens = 0;
            println!(" 🔥 STAGNATION! Increasing mutation to {:.3} for exploration", current_mut_rate);
            
            // Лёгкий каллинг для разнообразия
            let kill_count = (POP as f64 * 0.1) as usize;
            for i in (POP - kill_count)..POP {
                pop[i] = Genome::random();
            }
            println!(" ⚔️ Added {} random genomes for diversity", kill_count);
        }
        
        if pop[0].fit > best_fit {
            best_fit = pop[0].fit;
            let _ = File::create("best_bio.txt").and_then(|mut f| f.write_all(pop[0].seq.as_bytes()));
            println!(" 🏆 BEST: {:.0} (food:{} apples, neurons:{}, synapses:{}) 🏆", 
                     best_fit, max_food_in_gen, pop[0].brain.as_ref().unwrap().neurons.len(),
                     pop[0].brain.as_ref().unwrap().syns.len());
            stag_cnt = 0; 
            hall.push(pop[0].clone()); 
            if hall.len() > CHAMP_BANK { hall.remove(0); }
        } else { stag_cnt += 1; }
        
        let avg = pop.iter().map(|g| g.fit).sum::<f64>() / POP as f64;
        let tb = pop[0].brain.as_ref().unwrap();
        println!("Gen {} | Best: {:.0} | MaxFood: {} | Avg: {:.0} | N:{} | S:{} | Death:{} | Mut:{:.3} | Hold:{}", 
            gen, pop[0].fit, max_food_in_gen, avg, tb.neurons.len(), tb.syns.len(), pop[0].reason, 
            current_mut_rate, record_holding_gens);
        
        if gen % 30 == 0 && max_food_in_gen > 5 {
            let mut b = pop[0].brain.clone().unwrap(); 
            let mut sn = Snake::new();
            println!("\n=== DEMO (mut rate: {:.3}) ===", current_mut_rate);
            for _ in 0..1000 { 
                let head = *sn.body.front().unwrap();
                let sv = sn.sense(); 
                let dec = b.tick(&sv, &sn.body, sn.body.len(), sn.eaten, sn.turns, sn.loop_det, head, sn.food); 
                sn.update(dec); 
                render(&b, &sn, gen as usize, best_fit, sn.eaten, current_mut_rate); 
                if !sn.alive { 
                    println!("Demo finished: {} | Food: {}", sn.reason, sn.eaten);
                    sleep(Duration::from_secs(1));
                    break; 
                } 
                sleep(Duration::from_millis(15)); 
            }
        }
        
        let stagn = stag_cnt > STAG_LIM;
        let plateau = pop[0].brain.as_ref().map(|b| b.plateau_counter).unwrap_or(0);
        
        let mut new_pop: Vec<Genome> = Vec::new();
        
        for i in 0..ELITE { 
            new_pop.push(pop[i].clone()); 
        }
        
        let mut rng = rand::thread_rng();
        
        while new_pop.len() < POP {
            let p1_idx = if rng.gen_bool(0.25) && pop.len() > POP/2 {
                rng.gen_range(POP/2..POP)
            } else {
                rng.gen_range(0..POP/2)
            };
            let p2_idx = rng.gen_range(0..POP/2);
            
            let p1 = &pop[p1_idx];
            let p2 = &pop[p2_idx];
            
            let mut child = Genome::crossover(p1, p2, &pop[0].reason, gen, &pop[0].stats, stagn, max_food_in_gen, plateau, current_mut_rate);
            
            if rng.gen_bool(0.15) {
                if let Some(ref mut b) = child.brain {
                    b.deep_mutate("extra", gen, &pop[0].stats, true, max_food_in_gen, plateau, current_mut_rate);
                }
                child.seq = child.brain.as_ref().unwrap().to_genome();
            }
            new_pop.push(child);
        }
        
        if stag_cnt > STAG_LIM * 2 && !hall.is_empty() && current_mut_rate == HIGH_MUTATION_RATE {
            let champ_idx = rng.gen_range(0..hall.len());
            let champ = hall.iter().nth(champ_idx).unwrap().clone();
            for i in ELITE..(ELITE + 15).min(POP) {
                let mut mutated_champ = champ.clone();
                if let Some(ref mut b) = mutated_champ.brain {
                    b.deep_mutate("recovery", gen, &pop[0].stats, true, max_food_in_gen, plateau, current_mut_rate);
                }
                mutated_champ.seq = mutated_champ.brain.as_ref().unwrap().to_genome();
                new_pop[i] = mutated_champ;
            }
            stag_cnt = 0;
            println!(" 🔥 STAGNATION RECOVERY! 🔥");
        }
        
        pop = new_pop; 
        gen += 1;
        
        if gen % 100 == 0 {
            println!("💾 Saving checkpoint...");
            let _ = File::create("checkpoint.txt").and_then(|mut f| f.write_all(pop[0].seq.as_bytes()));
        }
    }
}