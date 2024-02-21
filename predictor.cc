#include "predictor.h"


#define CTR_MAX  3
#define CTR_INIT 2


PREDICTOR::PREDICTOR(void){

    std::fill_n(HybridTable, 1 << index_bits_hybrid, CTR_INIT);
    
    GHR = 0x0;
    
    _neural = new NeuralPredictor();
    _tage = new TagePredictor();
    
    PCmask_hybrid = (0x3fffffff >> (30 - index_bits_hybrid));
    
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

PREDICTOR::~PREDICTOR(void){
    delete _neural;
    delete _tage;
}

bool PREDICTOR::GetPrediction(UINT32 PC){
    prediction_neural = _neural->GetPrediction(PC);
    prediction_tage = _tage->GetPrediction(PC);
    
    int index = (PC >> 2 ^ GHR) & PCmask_hybrid;
  //  return prediction_tage;
    return HybridTable[index] > 1 ? prediction_tage : prediction_neural;
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void PREDICTOR::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
    int index = (PC >> 2 ^ GHR) & PCmask_hybrid;
    
    _tage->UpdatePredictor(PC, resolveDir, prediction_tage, branchTarget);

    _neural->UpdatePredictor(PC, resolveDir, prediction_neural, branchTarget);

    index = (PC >> 2 ^ GHR) & PCmask_hybrid;
    
    if (prediction_tage == resolveDir && prediction_neural != resolveDir)
    {
        SatIncrement(HybridTable[index], CTR_MAX);
    }
    if (prediction_tage != resolveDir && prediction_neural == resolveDir)
    {
        SatDecrement(HybridTable[index]);
    }
    
    // Update the global history record
    GHR = (GHR << 1) + (resolveDir ? 1 : 0);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void PREDICTOR::TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget){
    
    // This function is called for instructions which are not
    // conditional branches, just in case someone decides to design
    // a predictor that uses information from such instructions.
    // We expect most contestants to leave this function untouched.
    
    return;
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

NeuralPredictor::NeuralPredictor(void){
    // The theta value used as the threshold for determining weight saturation.
    kTheta = static_cast<int>(1.93 * kHistorySize + 14);
    // The number of bits each weight can use.
    kWeightSize = 1 + log2(kTheta);
    
    w_max = (1 << (kWeightSize - 1)) - 1;
    w_min = -(1 << (kWeightSize - 1));
    
    _bias = new int8_t[kSize];
    _weights = new int8_t[kSize*kHistorySize];
    GHR = 0x0;
}

NeuralPredictor::~NeuralPredictor(void){
    delete []_bias;
    delete []_weights;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool NeuralPredictor::GetPrediction(UINT32 PC){
    int key = get_key(PC);
    int y = _bias[key];
    for (int i = 0; i < kHistorySize; ++i) {
        int xi = (((GHR >> i) & 0x1) == 1) ? 1 : -1;
        y += _weights[key*kHistorySize+i] * xi;
    }
    _y_out = y;
    return y >= 0;
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  NeuralPredictor::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
    int t = resolveDir ? 1 : -1;
    int key = get_key(PC);
    if (sign(_y_out) != t || abs(_y_out) <= kTheta) {
        for (int i = 0; i < kHistorySize; ++i) {
            int xi = (((GHR >> i) & 0x1) == 1) ? 1 : -1;
            int wi = _weights[key*kHistorySize+i] + t * xi;
            if (wi >= w_min && wi <= w_max) {
                _weights[key*kHistorySize+i] = wi;
            }
        }
        int b = _bias[key] + t;
        if (b >= w_min && b <= w_max) {
            _bias[key] = b;
        }
    }
    GHR = ((GHR << 1) + ((t == 1)? 1 : 0));
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

TagePredictor::TagePredictor(void){
    lens[0] = MAX_LENGTH - 1;
    lens[N_BANKS - 1] = MIN_LENGTH;
    for (int i = 1; i < N_BANKS - 1; i ++) {
        double temp = pow((double)(MAX_LENGTH - 1) / MIN_LENGTH, (double)i / (N_BANKS - 1));
        lens[N_BANKS - i - 1] = (int) (MIN_LENGTH * temp + 0.5);
    }
    for (int i = 0; i < N_BANKS; i ++) {
        comp_hist_i[i].create(lens[i], LOG_GLOBAL);
        comp_hist_t[0][i].create(comp_hist_i[i].ORIGIN_LEN, TAG_BITS - ((i + (N_BANKS & 1)) / 2));
        comp_hist_t[1][i].create(comp_hist_i[i].ORIGIN_LEN, TAG_BITS - ((i + (N_BANKS & 1)) / 2) - 1);
    }
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool TagePredictor::GetPrediction(UINT32 PC){
    
    calc_index(PC);
    linear_search(PC);
    if (bank == N_BANKS)
        return alt_pred = get_base_pred(PC);
    if (alt_bank == N_BANKS)
        alt_pred = get_base_pred(PC);
    else
        alt_pred = (global_table[alt_bank][G_INDEX[alt_bank]].ctr >= 0);
    if (pred_store < 0 || abs(2 * global_table[bank][G_INDEX[bank]].ctr + 1) != 1 || global_table[bank][G_INDEX[bank]].ubit != 0)
        return global_table[bank][G_INDEX[bank]].ctr >= 0;
    return alt_pred;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void TagePredictor::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
    bool alloc = (predDir != resolveDir) & (bank > 0);
    if (bank < N_BANKS) {
        bool loc_taken = global_table[bank][G_INDEX[bank]].ctr >= 0;
        bool pseudo_new_alloc = (abs(2 * global_table[bank][G_INDEX[bank]].ctr + 1) == 1) && (global_table[bank][G_INDEX[bank]].ubit == 0);
        if (pseudo_new_alloc) {
            if (loc_taken == resolveDir)
                alloc = false;
            if (loc_taken != alt_pred) {
                if (alt_pred == resolveDir) {
                    if (pred_store < 7)
                        pred_store ++;
                }
                else {
                    if (pred_store > -8)
                        pred_store--;
                }
            }
        }
    }
    if (alloc)
        alloc_new_hist(resolveDir, PC);
    if (bank == N_BANKS)
        update_base(PC, resolveDir);
    else
        update_ctr(global_table[bank][G_INDEX[bank]].ctr, resolveDir, CTR_BITS);
    if (predDir != alt_pred && bank < N_BANKS) {
        if (predDir == resolveDir) {
            if (global_table[bank][G_INDEX[bank]].ubit < 3)
                global_table[bank][G_INDEX[bank]].ubit ++;
        }
        else {
            if (global_table[bank][G_INDEX[bank]].ubit > 0)
                global_table[bank][G_INDEX[bank]].ubit --;
        }
    }
    update_hist(resolveDir, PC);
}