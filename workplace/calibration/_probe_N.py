"""抽查 N 扫的原始数字：单边内缩的符号、假阳/漏、以及和体积比的对照。"""
import json

d = json.load(open("/home/zjq/lerobot/workplace/calibration/calib/recon/validate/runs_quick.json"))
runs = d["runs"]
rs = [r for r in runs if r["tag"].startswith("N|sts3215_03a_v1|")]
rs += [r for r in runs if r["tag"] == "marg|sts3215_03a_v1|0.0"]
rs.sort(key=lambda r: -r["theta_c"])
hdr = ("tag", "N", "th_c", "inset", "insetP", "假阳%", "漏%", "iou", "V_R/V_G")
print("%-26s %4s %6s %7s %7s %6s %6s %6s %8s" % hdr)
for r in rs:
    print("%-26s %4d %6.3f %+7.3f %+7.3f %6.2f %6.2f %6.3f %8.3f" % (
        r["tag"], r["n_views"], r["theta_c"], r["aud_inset_px"], r["aud_inset_perim_px"],
        r["false_pos"] * 100, r["miss_vol"] * 100, r["aud_iou"], r["vol_ratio"]))
