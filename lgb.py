# 对经纬度进行geohash编码
def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)
 
# tfidf
def tfidf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp
    
# count2vec
def count2vec(input_values, output_num, output_prefix, seed=1024):
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp
    
    
# workers设为1可复现训练好的词向量，但速度稍慢，若不考虑复现的话，可对此参数进行调整
def w2v_feat(df, group_id, feat, length):
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1,
                     workers=1, iter=10, seed=1, hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame
