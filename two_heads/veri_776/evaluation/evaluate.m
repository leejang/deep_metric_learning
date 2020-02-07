nQuery = 1678;
nTest = 11579;

% dimensions of the embedding feature vector
%nFeature = 512;
nFeature = 128;
%nFeature = 3;

%{

% mobilenet_v1 + softmax loss + random sampler + 128 embed_dim
queryfeature = h5read('mobilenet_v1_w_softmax_loss_random_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_softmax_loss_random_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_softmax_loss_random_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_softmax_loss_random_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss + pk sampler + 128 embed_dim
queryfeature = h5read('mobilenet_v1_w_cos_loss_pk_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_pk_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_pk_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_pk_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss + pk sampler + 128 embed_dim + w/ warmup
queryfeature = h5read('mobilenet_v1_w_cos_loss_pk_warmup_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_pk_warmup_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_pk_warmup_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_pk_warmup_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss + random sampler + 128 embed_dim + w/ warmup
queryfeature = h5read('mobilenet_v1_w_cos_loss_random_warmup_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_random_warmup_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_random_warmup_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_random_warmup_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss + pk sampler + 128 embed_dim + init_lr 0.05
queryfeature = h5read('mobilenet_v1_w_cos_loss_0.05_pk_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_0.05_pk_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_0.05_pk_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_0.05_pk_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss + random sampler + 3 embed_dim
queryfeature = h5read('mobilenet_v1_w_cos_loss_random_emb3_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_random_emb3_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_random_emb3_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_random_emb3_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + softmax loss + random sampler + 3 embed_dim
queryfeature = h5read('mobilenet_v1_w_softmax_loss_random_emb3_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_softmax_loss_random_emb3_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_softmax_loss_random_emb3_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_softmax_loss_random_emb3_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss (s:30, m:0.35)  + pk sampler + 512 embed_dim + adam
queryfeature = h5read('mobilenet_v1_w_cos_loss_emb_512_pk_adam_s30_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_emb_512_pk_adam_s30_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_emb_512_pk_adam_s30_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_emb_512_pk_adam_s30_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss (s:30, m:0.35)  + pk sampler + 128 embed_dim + adam
queryfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s30_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s30_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s30_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s30_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss (s:64, m:0.35)  + pk sampler (30x5) + 128 embed_dim + adam
queryfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_30_5_adam_s64_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_30_5_adam_s64_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_30_5_adam_s64_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_30_5_adam_s64_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + cos loss (s:64, m:0.35)  + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model
queryfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s64_image_query_features.h5', '/Emb_vectors');
testfeature = h5read('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s64_image_test_features.h5', '/Emb_vectors');

h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s64_image_query_features.h5', '/Emb_vectors');
h5disp('mobilenet_v1_w_cos_loss_emb_128_pk_adam_s64_image_test_features.h5', '/Emb_vectors');

% mobilenet_v1 + triplet (BH) + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model
queryfeature = h5read('veri776_query_embeddings.h5', '/emb');
testfeature = h5read('veri776_test_embeddings.h5', '/emb');

h5disp('veri776_query_embeddings.h5', '/emb');
h5disp('veri776_test_embeddings.h5', '/emb');

% mobilenet_v1 + lmcl + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model
queryfeature = h5read('veri776_query_embeddings_lmcl.h5', '/emb');
testfeature = h5read('veri776_test_embeddings_lmcl.h5', '/emb');

h5disp('veri776_query_embeddings_lmcl.h5', '/emb');
h5disp('veri776_test_embeddings_lmcl.h5', '/emb');

% mobilenet_v1 + lmcl + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model + BH (Batch Hard)
queryfeature = h5read('veri776_query_embeddings_lmcl_bh.h5', '/emb');
testfeature = h5read('veri776_test_embeddings_lmcl_bh.h5', '/emb');

h5disp('veri776_query_embeddings_lmcl_bh.h5', '/emb');
h5disp('veri776_test_embeddings_lmcl_bh.h5', '/emb');

% mobilenet_v1 + triplet cosine + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model
queryfeature = h5read('veri776_query_embeddings_triplet_cosine.h5', '/emb');
testfeature = h5read('veri776_test_embeddings_triplet_cosine.h5', '/emb');

h5disp('veri776_query_embeddings_triplet_cosine.h5', '/emb');
h5disp('veri776_test_embeddings_triplet_cosine.h5', '/emb');

queryfeature = reshape(queryfeature, [nFeature, nQuery]);
testfeature = reshape(testfeature, [nFeature, nTest]);
%}

% mobilenet_v1 + two losses + pk sampler + 128 embed_dim + adam + fine-tuning with imagenet based trained model
queryfeature_1 = h5read('veri776_query_embeddings_two_losses_1.h5', '/emb');
testfeature_1 = h5read('veri776_test_embeddings_two_losses_1.h5', '/emb');

h5disp('veri776_query_embeddings_two_losses_1.h5', '/emb');
h5disp('veri776_test_embeddings_two_losses_1.h5', '/emb');

queryfeature_2 = h5read('veri776_query_embeddings_two_losses_2.h5', '/emb_2');
testfeature_2 = h5read('veri776_test_embeddings_two_losses_2.h5', '/emb_2');

h5disp('veri776_query_embeddings_two_losses_2.h5', '/emb_2');
h5disp('veri776_test_embeddings_two_losses_2.h5', '/emb_2');

queryfeature_1 = reshape(queryfeature_1, [nFeature, nQuery]);
testfeature_1 = reshape(testfeature_1, [nFeature, nTest]);

queryfeature_2 = reshape(queryfeature_2, [nFeature, nQuery]);
testfeature_2 = reshape(testfeature_2, [nFeature, nTest]);

queryfeature_1 = norm_code(queryfeature_1);
testfeature_1 = norm_code(testfeature_1);

queryfeature_2 = norm_code(queryfeature_2);
testfeature_2 = norm_code(testfeature_2);

dist_1 = 1 - testfeature_1' * queryfeature_1;
dist_2 = 1 - testfeature_2' * queryfeature_2;

dist = dist_1 + dist_2;
%dist = pdist2(testfeature', queryfeature', 'cosine');

%{
dist = zeros(nTest, nQuery);
for i = 1:nTest
  for j = 1:nQuery
    X1 = testfeature(:,i);
    X2 = queryfeature(:,j);
    dist(i, j) = 1 - getCosineSimilarity(X1, X2);
  end
end
% check size (debug)
%size(queryfeature)
%size(testfeature)
%size(dist)
%}

maxgt = 256;
gt_index =  zeros(nQuery, maxgt);
fidin = fopen('gt_index.txt');

for i = 1:nQuery
    gt_index_line = fgetl(fidin);
    gt_line = str2num(gt_index_line);
    for j = 1:size(gt_line, 2)
       gt_index(i, j) = gt_line(j); 
    end
end

maxjk = 256;
jk_index = zeros(nQuery, maxjk);
fidin = fopen('jk_index.txt');
for i = 1:nQuery
    jk_index_line = fgetl(fidin);
    jk_line = str2num(jk_index_line);
    for j = 1:size(jk_line, 2)
       jk_index(i, j) = jk_line(j); 
    end
end

ap = zeros(nQuery, 1); % average precision
CMC = zeros(nQuery, nTest);
r1 = 0; % rank 1 precision with single query
parfor k = 1:nQuery
      %k
      good_index = reshape(gt_index(k,:), 1, []);
      good_index = good_index(good_index ~= 0);
      junk_index = reshape(jk_index(k,:), 1, []);
      junk_index = junk_index(junk_index ~= 0);
    score = dist(:, k);
    [~, index] = sort(score, 'ascend');  % single query

    [ap(k), CMC(k, :)] = ... % compute AP for single query
        compute_AP(good_index, junk_index, index);

end
CMC = mean(CMC);
fprintf('new model: mAP = %f,\nr1 = %f,\nr5 = %f\r\n',...
    mean(ap), CMC(1), CMC(5));
% figure;
%s = 50;
%CMC_curve = CMC;
% plot(1:s, CMC_curve(:, 1:s));
% toc;

clear k ap fidin gt_index gt_index_line
clear gt_line i j nTest nQuery r1 maxgt maxjk
clear jk_index jk_index_line jk_line CMC_curve
clear testfeature queryfeature
disp('done!');


