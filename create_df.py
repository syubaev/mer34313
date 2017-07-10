def add_cluster_X0_to_df(df):
    """
    Присваевает кластер категории X0, по среднему значению y в трейне.
    Кодирует кластера с помощью OHE
    """
    CLUST_NUMB = 9
    CLUST_MIN_CNT = 65
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y = train.y
    le = LabelEncoder().fit(pd.concat([train.X0, test.X0]))

    # Кодируем X0 в трейне и тесте
    X0_le, test_X0_le = le.transform(train.X0), le.transform(test.X0)

    ohe = OneHotEncoder()
    X0_ohe = ohe.fit_transform(X0_le.reshape(-1,1))

    # Словарь соответствия кода X0 номеру коэф в регрессии
    d_le_coef = dict([(le,i) for i, le in enumerate(ohe.active_features_)], )

    # Вычисляем регрессию для всех кодов
    model = LinearRegression().fit(X0_ohe, y)
    kmeans = KMeans(n_clusters=CLUST_NUMB)
    coef_clust = kmeans.fit_predict(model.coef_.reshape(-1,1))

    # Найти код Х0 в списке коэфицинтов по кластерам
    X0_clust = np.array([coef_clust[ d_le_coef.get(x0_code) ] for x0_code in X0_le])

    # Если кода нету в трейне и для него не рассчитан коэфициент, то присвоить значение самого массового кластера
    POPULAR_CLASTER = pd.DataFrame({'cl':X0_clust}).cl.value_counts().index[0]
    test_X0_clust = np.array(
        [coef_clust[ d_le_coef.get(x0_code, POPULAR_CLASTER) ] 
                              for x0_code in test_X0_le])
    # Объеденяем кластеры трейна и теста.
    clust_label = np.vstack((X0_clust.reshape(-1,1), test_X0_clust.reshape(-1,1)))

    unique, counts = np.unique(clust_label, return_counts=True)
    while np.any(counts < CLUST_MIN_CNT):
        ind = np.argmin(counts)
        cl_small = unique[ind]
        cl_center = kmeans.cluster_centers_.flatten()[ind]
        # Убираем выбранный кластер из общего списка
        mask = np.ones(unique.shape,dtype=bool); mask[ind] = False

        centers_arr = kmeans.cluster_centers_.flatten()[mask]
        cl_nearest = unique[mask][np.argmin( np.abs(centers_arr - cl_center) )]
        print(cl_small, 'to', cl_nearest)
        # Обновляем список кластеров
        clust_label[clust_label == cl_small] = cl_nearest
        unique, counts = np.unique(clust_label, return_counts=True)

    
    X0_ohe, test_X0_ohe = ohe.fit_transform(clust_label[ix_train]),\
        ohe.transform(clust_label[ix_test])

    feat_clust = ['clust_' + str(i) for i in np.arange(X0_ohe.shape[1])]
    t = pd.DataFrame(vstack((X0_ohe, test_X0_ohe)).todense(), columns=feat_clust)
    
    df['clust_label'] = clust_label
    df = pd.concat([df,t], axis=1)
    return df, feat_clust


def add_OHE_for_categ(df):
    """
    Кодирует категориальные в OHE
    """
    for col in feat_categ:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = OneHotEncoder().fit_transform(df[feat_categ])

    feat_ohe = ['ohe_' + str(i) for i in np.arange(X.shape[1])]
    t = pd.DataFrame(X.todense(), columns=feat_ohe, index=df.index)
    df = pd.concat( [df, t] , axis=1)
    return df, feat_ohe


def add_PCA_ICA(df):
    """
    Добавляет PCA ICA преобразования для числовых и категориальных после OHE
    """
    feat_list = feat_numb + d_feat['feat_ohe']
    n_comp = 20

    # PCA
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(df[ix_train][feat_list])
    pca2_df = pca.transform(df[feat_list])

    # ICA
    ica = FastICA(n_components=n_comp, random_state=42, max_iter = 600)
    ica.fit(df[ix_train][feat_list])
    ica2_df= ica.transform(df[feat_list])

    # Append decomposition components to datasets
    feat_pca = list()
    feat_ica = list()
    for i in range(1, n_comp+1):
        df['pca_' + str(i)] = pca2_df[:,i-1]
        df['ica_' + str(i)] = ica2_df[:,i-1]
        feat_pca.append('pca_' + str(i))
        feat_ica.append('ica_' + str(i))
        
    return df, feat_pca, feat_ica

def add_some_pairs(df):
    df, d_feat['feat_pca'], d_feat['feat_ica'] = add_PCA_ICA(df)
    # Добавление пар
    with open('./../tmp/model_pairs_list.pkl', 'rb') as f:
        model_pairs_list = pickle.load(f)
    with open('./../tmp/model_score_list.pkl', 'rb') as f:
        model_score_list = pickle.load(f)

    def woker_split(p_in):
        p = p_in.split("_")
        f1 = p[0]
        f2 = p[1]
        df[p_in] = df[f1] * df[f2]
        return p_in
    top_index = pd.DataFrame({'scr':model_score_list}).\
        sort_values('scr', ascending=False).\
          head(5).index

    feat_pairs_in_top = sum([model_pairs_list[ind] for ind in top_index], [])
    feat_pairs_in_top = list(set(feat_pairs_in_top))

    _ = [woker_split(p) for p in feat_pairs_in_top]
    return df, feat_pairs_in_top


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train.y

# train delete outlire
outlier_filter = y < np.percentile(y, 99.5)
train = train[outlier_filter].reset_index(drop=True)
y = train.y

feat_all = train.columns[2:]
feat_categ = train.dtypes.index[train.dtypes == 'object']
feat_not_uniq = ['X289',
 'X235',
 'X347',
 'X268',
 'X11',
 'X107',
 'X233',
 'X330',
 'X290',
 'X293',
 'X297',
 'X93']

feat_numb = list(set(feat_all) - set(feat_categ) - set(feat_not_uniq))

train['test'] = 0
test['test'] = 1

d_feat = {}
    
df = pd.concat([train, test], axis=0).reset_index(drop=True)
df = df.drop(["ID", "y"], axis=1)
ix_train = df.test == 0
ix_test = df.test == 1


print('Добавляем OneHotEncoder для категориальных фич')
df, d_feat['feat_ohe'] = add_OHE_for_categ(df)

print('Добавляем PCA ICA')
df, d_feat['feat_pca'], d_feat['feat_ica']= add_PCA_ICA(df)

print('Добавляем пары')
df, d_feat['feat_pairs_in_top'] = add_some_pairs(df)

print('Добавляем Кластарезацию по X0')
df, d_feat['feat_clust'] = add_cluster_X0_to_df(df)

print('train.shape=', train.shape, 'df shape = ', df.shape)

for key,val in d_feat.items():
    exec(key + '=val')
    print(key)
