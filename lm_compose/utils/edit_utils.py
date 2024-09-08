# utils for TAXI dataset
def make_edit_batches(df):
    df2 = df.copy()
    batches = []
    while df2.shape[0] > 0:
        batch = df2.groupby(["entity"]).sample(1)
        batches.append(batch)
        df2 = df2.loc[lambda x: ~x.edit.isin(batch.edit)]

    return batches


def make_rewrite(e):
    rewrite = {
        "prompt": f"A {e.subj} is a kind of",
        "target_new": e.entity,  # {'str': e.entity},
        "subject": e.subj,
    }

    return rewrite
