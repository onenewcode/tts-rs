use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

pub(crate) fn normalized_codebook_centroids<B: Backend>(
    cluster_usage: Tensor<B, 1>,
    embedding_sum: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [codebook_size, _embed_dim] = embedding_sum.dims();
    embedding_sum.div(cluster_usage.clamp_min(1e-6).reshape([codebook_size, 1]))
}

pub(crate) fn nearest_codebook_token_ids<B: Backend>(
    hidden: Tensor<B, 3>,
    centroids: Tensor<B, 2>,
) -> Tensor<B, 2, Int> {
    let [batch_size, hidden_size, time_steps] = hidden.dims();
    let [codebook_size, codebook_hidden] = centroids.dims();
    assert_eq!(
        hidden_size, codebook_hidden,
        "hidden/codebook embedding size mismatch"
    );

    let distances = (hidden.swap_dims(1, 2).unsqueeze_dim::<4>(2)
        - centroids.reshape([1, 1, codebook_size, codebook_hidden]))
    .square()
    .sum_dim(3)
    .squeeze_dim::<3>(3);
    distances.argmin(2).reshape([batch_size, time_steps])
}

pub(crate) fn gather_codebook_embeddings<B: Backend>(
    codebook: Tensor<B, 2>,
    token_ids: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let [batch, seq_len] = token_ids.dims();
    let [_codebook_size, embed_dim] = codebook.dims();
    codebook
        .select(0, token_ids.reshape([batch * seq_len]))
        .reshape([batch, seq_len, embed_dim])
        .swap_dims(1, 2)
}
