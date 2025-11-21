import argparse
from collections import Counter, defaultdict
from pprint import pprint

def load_data():
    return [
        {"outlook":"Sunny","humidity":"High","wind":"Weak","play":"No"},
        {"outlook":"Sunny","humidity":"High","wind":"Strong","play":"No"},
        {"outlook":"Overcast","humidity":"High","wind":"Weak","play":"Yes"},
        {"outlook":"Rain","humidity":"High","wind":"Weak","play":"Yes"},
        {"outlook":"Rain","humidity":"Normal","wind":"Weak","play":"Yes"},
        {"outlook":"Rain","humidity":"Normal","wind":"Strong","play":"No"},
        {"outlook":"Overcast","humidity":"Normal","wind":"Strong","play":"Yes"},
        {"outlook":"Sunny","humidity":"High","wind":"Weak","play":"No"},
        {"outlook":"Sunny","humidity":"Normal","wind":"Weak","play":"Yes"},
        {"outlook":"Rain","humidity":"High","wind":"Weak","play":"Yes"},
        {"outlook":"Sunny","humidity":"Normal","wind":"Strong","play":"Yes"},
        {"outlook":"Overcast","humidity":"High","wind":"Strong","play":"Yes"},
        {"outlook":"Overcast","humidity":"Normal","wind":"Weak","play":"Yes"},
        {"outlook":"Rain","humidity":"High","wind":"Strong","play":"No"},
    ]

def make_freq_tables(rows, feats, target="play"):
    cls_cnt = Counter(r[target] for r in rows)

    feat_cnts = {f: {cl: Counter() for cl in cls_cnt} for f in feats}
    feat_vals = {f: set() for f in feats}

    for r in rows:
        cl = r[target]
        for f in feats:
            v = r[f]
            feat_cnts[f][cl][v] += 1
            feat_vals[f].add(v)

    feat_vals = {f: sorted(list(vs)) for f, vs in feat_vals.items()}
    return cls_cnt, feat_cnts, feat_vals


def make_like_tables(cls_cnt, feat_cnts, feat_vals, alpha=0):
    like_tbl = defaultdict(lambda: defaultdict(dict))
    for f, per_class in feat_cnts.items():
        k = len(feat_vals[f])
        for cl, cnts in per_class.items():
            total = cls_cnt[cl]
            for v in feat_vals[f]:
                like_tbl[f][cl][v] = (cnts.get(v, 0) + alpha) / (total + alpha * k)
    return like_tbl

def get_posterior(x, cls_cnt, like_tbl):
    total_n = sum(cls_cnt.values())
    probs = {}

    for cl, cnum in cls_cnt.items():
        prior = cnum / total_n
        mul = 1
        for f, v in x.items():
            mul *= like_tbl[f][cl].get(v, 0)
        probs[cl] = prior * mul

    s = sum(probs.values())
    if s > 0:
        for cl in probs:
            probs[cl] /= s

    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0, help="лапласове згладжування (можна 1)")
    args = parser.parse_args()

    data = load_data()
    feats = ["outlook", "humidity", "wind"]

    print("\nЧастотні таблиці:")
    cls_cnt, feat_cnts, feat_vals = make_freq_tables(data, feats)

    print("\nКількість прикладів кожного класу:")
    pprint(cls_cnt)

    print("\nМожливі значення ознак:")
    pprint(feat_vals)

    print("\nЧастоти ознак по класах:")
    for f in feats:
        print(f"\nОзнака '{f}':")
        pprint(feat_cnts[f])

    print("\nТаблиці ймовірностей:")
    like_tbl = make_like_tables(cls_cnt, feat_cnts, feat_vals, alpha=args.alpha)

    for f in feats:
        print(f"\nЙмовірності P({f}=value | class):")
        pprint(dict(like_tbl[f]))

    print("\nАпостеріорні ймовірності:")
    x = {"outlook": "Rain", "humidity": "High", "wind": "Weak"}
    print("Перевіряємо приклад:", x)

    probs = get_posterior(x, cls_cnt, like_tbl)
    print("\nРезультат:")
    pprint(probs)

    ans = max(probs, key=probs.get)
    print("\nВисновок: матч буде?", ans)

if __name__ == "__main__":
    main()
