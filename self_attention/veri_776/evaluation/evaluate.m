nQuery = 1678;
nTest = 11579;
nFeature = 1024;

%{
% mobilenet_v1 with std softmax loss
queryfeature = h5read('mobilenet_v1_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with cos loss
queryfeature = h5read('mobilenet_v1_w_cos_loss_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_cos_loss_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_cos_loss_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_cos_loss_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with self-attention and std softmax loss
queryfeature = h5read('mobilenet_v1_w_sa_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_sa_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_sa_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_sa_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with self-attention and cos loss
queryfeature = h5read('mobilenet_v1_w_sa_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_sa_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_sa_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_sa_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with self-attention_56 and cos loss
queryfeature = h5read('mobilenet_v1_w_sa_56_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_sa_56_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_sa_56_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_sa_56_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with two self-attention + sn and cos loss
queryfeature = h5read('mobilenet_v1_w_sa_two_self_sn_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_sa_two_self_sn_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_sa_two_self_sn_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_sa_two_self_sn_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with two self-attention + cos loss + batch sampling
queryfeature = h5read('mobilenet_v1_w_cos_loss_batch_sample_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_cos_loss_batch_sample_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_cos_loss_batch_sample_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_cos_loss_batch_sample_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with two self-attention + sn and cos loss
queryfeature = h5read('mobilenet_v1_w_sa_two_self_w_max_pooling_sn_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_sa_two_self_w_max_pooling_sn_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_sa_two_self_w_max_pooling_sn_n_cos_loss_image_query_features.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_sa_two_self_w_max_pooling_sn_n_cos_loss_image_test_features.h5', '/AvgPool_1a');

% mobilenet_v1 with two self-attention + sn and cos loss + pk sampler + 1024 embed_dim
queryfeature = h5read('mobilenet_v1_w_cos_loss_batch_sample_image_query_features_1024.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_cos_loss_batch_sample_image_test_features_1024.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_cos_loss_batch_sample_image_query_features_1024.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_cos_loss_batch_sample_image_test_features_1024.h5', '/AvgPool_1a');
%}

% mobilenet_v1 with two self-attention + sn and cos loss + random sampler + 1024 embed_dim
queryfeature = h5read('mobilenet_v1_w_n_cos_loss_wo_batch_sample_image_query_features_1024.h5', '/AvgPool_1a');
testfeature = h5read('mobilenet_v1_w_n_cos_loss_wo_batch_sample_image_test_features_1024.h5', '/AvgPool_1a');

h5disp('mobilenet_v1_w_n_cos_loss_wo_batch_sample_image_query_features_1024.h5', '/AvgPool_1a');
h5disp('mobilenet_v1_w_n_cos_loss_wo_batch_sample_image_test_features_1024.h5', '/AvgPool_1a');

queryfeature = reshape(queryfeature, [nFeature, nQuery]);
testfeature = reshape(testfeature, [nFeature, nTest]);

queryfeature = norm_code(queryfeature);
testfeature = norm_code(testfeature);

%dist = 1 - testfeature' * queryfeature;

dist = pdist2(testfeature', queryfeature', 'cosine');

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
%     k
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function d = bhattacharyya(X1, X2)
  %check inputs and output
  error(nargchk(2,2,nargin));
  error(nargoutchk(0,1,nargout));

  [n,m]=size(X1);
  % check dimension 
  % assert(isequal(size(X2),[n m]),'Dimension of X1 and X2 mismatch.');
  assert(size(X2,2)==m,'Dimension of X1 and X2 mismatch.');

  mu1=mean(X1);
  C1=cov(X1);
  mu2=mean(X2);
  C2=cov(X2);
  C=(C1+C2)/2;
  dmu=(mu1-mu2)/chol(C);
  try
    d=0.125*dmu*dmu'+0.5*log(det(C/chol(C1*C2)));
  catch
    d=0.125*dmu*dmu'+0.5*log(abs(det(C/sqrtm(C1*C2))));
    warning('MATLAB:divideByZero','Data are almost linear dependent. The results may not be accurate.');
  end
end

function Cs = getCosineSimilarity(x,y)
% 
% call:
% 
%      Cs = getCosineSimilarity(x,y)
%      
% Compute Cosine Similarity between vectors x and y.
% x and y have to be of same length. The interpretation of 
% cosine similarity is analogous to that of a Pearson Correlation
% 
% R.G. Bettinardi
% -----------------------------------------------------------------
  if isvector(x)==0 || isvector(y)==0
    error('x and y have to be vectors!')
  end
  if length(x)~=length(y)
    error('x and y have to be same length!')
  end
  xy   = dot(x,y);
  nx   = norm(x);
  ny   = norm(y);
  nxny = nx*ny;
  Cs   = xy/nxny;
end
