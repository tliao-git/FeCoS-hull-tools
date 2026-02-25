#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# python plot_correlations.py processed_data.csv --outdir plots

def main():
    p = argparse.ArgumentParser()
    p.add_argument("processed_csv", help="CSV with Eform, Ehull, Js columns")
    p.add_argument("--outdir", default=".", help="output directory")
    args = p.parse_args()

    df = pd.read_csv(args.processed_csv)

    # exclude those index are not numbers
    df = df[pd.to_numeric(df['index'], errors='coerce').notna()]

    # 1) Formation energy vs Js (Js on x, Eform on y)
    plt.figure()
    plt.scatter(
        df["Js"], df["Eform"],
        facecolors='none', edgecolors='blue', linewidths=0.8
    )
    plt.axvline(1.0, color="grey", linestyle="--", label=r"$J_\mathrm{s}$ = 1 T")
    plt.axhline(0.0, color="grey", linestyle="--", label=r"$E_\mathrm{form}$ = 0")
    plt.xlabel(r"$J_\mathrm{s}$ (T)")
    plt.ylabel("Formation energy (eV/atom)")
    plt.title(r"$E_\mathrm{form}$ vs $J_\mathrm{s}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/Eform_vs_Js.png", dpi=300)
    plt.close()
    print("Saved Eform_vs_Js.png")

    # 2a) Energy above hull vs Js (full scale)
    plt.figure()
    plt.scatter(
        df["Js"], df["Ehull"],
        facecolors='none', edgecolors='green', linewidths=0.8
    )
    plt.axvline(1.0, color="grey", linestyle="--", label=r"$J_\mathrm{s}$ = 1 T")
    plt.axhline(0.1, color="grey", linestyle="--", label=r"$E_\mathrm{hull}$ = 0.1 eV/atom")
    plt.xlabel(r"$J_\mathrm{s}$ (T)")
    plt.ylabel("Energy above hull (eV/atom)")
    plt.title(r"$E_\mathrm{hull}$ vs $J_\mathrm{s}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/Ehull_vs_Js.png", dpi=300)
    plt.close()
    print("Saved Ehull_vs_Js.png")

    # 2b) Energy above hull vs Js (y-limit = 2 eV/atom)
    plt.figure()
    plt.scatter(
        df["Js"], df["Ehull"],
        facecolors='none', edgecolors='green', linewidths=0.8
    )
    plt.ylim(0, 2.0)
    plt.axvline(1.0, color="grey", linestyle="--", label=r"$J_\mathrm{s}$ = 1 T")
    plt.axhline(0.1, color="grey", linestyle="--", label=r"$E_\mathrm{hull}$ = 0.1 eV/atom")
    plt.xlabel(r"$J_\mathrm{s}$ (T)")
    plt.ylabel("Energy above hull (eV/atom)")
    plt.title(r"$E_\mathrm{hull}$ vs $J_\mathrm{s}$ (y ≤ 2 eV/atom)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/Ehull_vs_Js_ylim2.png", dpi=300)
    plt.close()
    print("Saved Ehull_vs_Js_ylim2.png")

if __name__ == "__main__":
    main()

