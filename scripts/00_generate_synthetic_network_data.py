#!/usr/bin/env python3
import argparse, random, math, csv
from pathlib import Path
from datetime import datetime, timedelta

LABELS = ['Benign','PortScan','DDoS','BruteForce','Exfiltration','C2Beacon']
DEFAULT_START = datetime(2026, 1, 1)

def ip(rng, private=True):
    if private:
        return f"10.10.{rng.randint(1,8)}.{rng.randint(2,250)}"
    return f"203.0.113.{rng.randint(2,250)}"

def row_for(label, ts, rng):
    proto = rng.choice(['TCP','UDP'])
    if label == 'Benign':
        duration=rng.expovariate(1/7)+0.2; sp=rng.randint(1024,65000); dp=rng.choice([53,80,123,443,993,22])
        packets=rng.randint(3,90); bytes_out=rng.randint(200,90000); bytes_in=rng.randint(200,140000); flags='SF'; alert_count=0; unique_ports=rng.randint(1,4); failed=0
    elif label == 'PortScan':
        duration=rng.uniform(0.01,1.5); sp=rng.randint(1024,65000); dp=rng.randint(1,65535)
        packets=rng.randint(1,5); bytes_out=rng.randint(40,400); bytes_in=rng.randint(0,200); flags=rng.choice(['S0','REJ']); alert_count=rng.randint(1,3); unique_ports=rng.randint(20,300); failed=rng.randint(5,80)
    elif label == 'DDoS':
        duration=rng.uniform(0.1,5); sp=rng.randint(1024,65000); dp=rng.choice([80,443,53])
        packets=rng.randint(500,8000); bytes_out=rng.randint(20000,2000000); bytes_in=rng.randint(0,5000); flags=rng.choice(['S0','SF']); alert_count=rng.randint(2,8); unique_ports=rng.randint(1,5); failed=rng.randint(0,3)
    elif label == 'BruteForce':
        duration=rng.uniform(2,80); sp=rng.randint(1024,65000); dp=rng.choice([22,3389,445,21,25])
        packets=rng.randint(30,500); bytes_out=rng.randint(5000,150000); bytes_in=rng.randint(1000,40000); flags=rng.choice(['REJ','S0','SF']); alert_count=rng.randint(1,5); unique_ports=rng.randint(1,3); failed=rng.randint(30,400)
    elif label == 'Exfiltration':
        duration=rng.uniform(20,600); sp=rng.randint(1024,65000); dp=rng.choice([443,22,8080,8443])
        packets=rng.randint(200,4000); bytes_out=rng.randint(2000000,40000000); bytes_in=rng.randint(5000,500000); flags='SF'; alert_count=rng.randint(0,3); unique_ports=rng.randint(1,6); failed=rng.randint(0,2)
    else: # C2Beacon
        duration=rng.uniform(0.05,3); sp=rng.randint(1024,65000); dp=rng.choice([443,80,53,8080])
        packets=rng.randint(3,30); bytes_out=rng.randint(100,3000); bytes_in=rng.randint(100,8000); flags='SF'; alert_count=rng.randint(0,2); unique_ports=rng.randint(1,3); failed=rng.randint(0,2)
    bps=(bytes_in+bytes_out)/max(duration,0.01)
    pps=packets/max(duration,0.01)
    out_ratio=bytes_out/max(bytes_in+bytes_out,1)
    entropy=rng.uniform(2.0,4.5) if label=='Benign' else rng.uniform(3.5,7.8)
    beacon_score=rng.uniform(0,0.2) if label!='C2Beacon' else rng.uniform(0.75,0.99)
    burst_score=min(1.0, math.log1p(pps)/8)
    return {
        'timestamp':ts.isoformat(), 'src_ip': ip(rng, True), 'dst_ip': ip(rng, False if label in ['Exfiltration','C2Beacon'] else True),
        'src_port':sp, 'dst_port':dp, 'proto':proto, 'duration':round(duration,4), 'packets':packets,
        'bytes_out':bytes_out, 'bytes_in':bytes_in, 'bytes_per_sec':round(bps,4), 'packets_per_sec':round(pps,4),
        'outbound_ratio':round(out_ratio,4), 'unique_dst_ports_5m':unique_ports, 'failed_conn_5m':failed,
        'dns_entropy':round(entropy,4), 'beacon_score':round(beacon_score,4), 'burst_score':round(burst_score,4),
        'suricata_alert_count':alert_count, 'tcp_state':flags, 'asset_criticality': rng.choice([1,2,2,3,4,5]), 'label':label
    }

def generate_dataset(rows, output, seed=42):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    weights=[0.62,0.10,0.08,0.08,0.06,0.06]
    ts=DEFAULT_START
    with output.open('w', newline='') as f:
        writer=None
        for _ in range(rows):
            label=rng.choices(LABELS,weights=weights)[0]
            ts += timedelta(seconds=rng.randint(1,7))
            row=row_for(label,ts,rng)
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row.keys()),lineterminator='\n'); writer.writeheader()
            writer.writerow(row)
    return output

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rows',type=int,default=12000); ap.add_argument('--out',default='data/synthetic_flows.csv'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    generate_dataset(args.rows, args.out, args.seed)
    print(f'[+] wrote {args.rows} rows to {args.out}')
if __name__=='__main__': main()
