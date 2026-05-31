use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

pub(crate) fn normalized_codebook_centroids<B: Backend>(
    cluster_usage: Tensor<B, 1>,
    embedding_sum: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [codebook_size, _embed_dim] = embedding_sum.dims();
    let usage = cluster_usage
        .clamp_min(1e-6)
        .reshape([codebook_size, 1])
        .cast(DType::F32);
    embedding_sum.cast(DType::F32).div(usage)
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

    let distances = (hidden
        .swap_dims(1, 2)
        .cast(DType::F32)
        .unsqueeze_dim::<4>(2)
        - centroids
            .clone()
            .reshape([1, 1, codebook_size, codebook_hidden]))
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

#[cfg(test)]
mod tests {
    use super::{
        gather_codebook_embeddings, nearest_codebook_token_ids, normalized_codebook_centroids,
    };
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor, TensorData};

    #[test]
    fn normalized_codebook_centroids_divides_embedding_sum_by_usage() {
        let device = Default::default();
        let usage = Tensor::<Flex, 1>::from_data(TensorData::from([2.0_f32, 4.0]), &device);
        let embedding_sum = Tensor::<Flex, 2>::from_data(
            TensorData::new(vec![2.0_f32, 4.0, 12.0, 20.0], [2, 2]),
            &device,
        );

        let centroids = normalized_codebook_centroids(usage, embedding_sum);
        let values = centroids
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("centroids should be readable");

        assert_eq!(values, vec![1.0, 2.0, 3.0, 5.0]);
    }

    #[test]
    fn nearest_codebook_token_ids_picks_smallest_l2_centroid() {
        let device = Default::default();
        let hidden = Tensor::<Flex, 3>::from_data(
            TensorData::new(vec![1.0_f32, 4.0, 2.0, 5.0], [1, 2, 2]),
            &device,
        );
        let codebook = Tensor::<Flex, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 4.0, 5.0, 8.0, 9.0], [3, 2]),
            &device,
        );

        let token_ids = nearest_codebook_token_ids(hidden, codebook);
        let values = token_ids
            .into_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("token ids should be readable");

        assert_eq!(values, vec![0, 1]);
    }

    #[test]
    fn gather_codebook_embeddings_restores_batch_channel_time_layout() {
        let device = Default::default();
        let codebook = Tensor::<Flex, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 10.0, 20.0, 100.0, 200.0], [3, 2]),
            &device,
        );
        let token_ids = Tensor::<Flex, 2, Int>::from_data(
            TensorData::new(vec![2_i64, 0_i64, 1_i64], [1, 3]),
            &device,
        );

        let gathered = gather_codebook_embeddings(codebook, token_ids);
        let values = gathered
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gathered embeddings should be readable");

        assert_eq!(values, vec![100.0, 1.0, 10.0, 200.0, 2.0, 20.0]);
    }
}
