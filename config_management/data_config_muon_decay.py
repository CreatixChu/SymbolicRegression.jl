
class DataConfig:
    data_file_path = "./data/MuonDecay.csv"
    processed_data_prefix = "muon_decay"
    columns = ['m12^2', 'm23^2']
    b_adj = 0.002 # Bandwidth adjustment for KDE adjustment factor on scott's factor for the bandwidth
    K = 'tophat' # Kernel type for KDE
    slice_num = 8 # Number of slices for conditional distributions