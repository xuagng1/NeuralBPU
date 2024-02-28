#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
#include "tracer.h"
#include <cmath>
#include <cstdlib>
#include <stdint.h>
#include <bitset>
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
class NeuralPredictor;
class TagePredictor;

class PREDICTOR{
private:
    UINT64 GHR;           // global history register
    static const UINT32 index_bits_hybrid = 10;
    
    uint8_t HybridTable[1 << index_bits_hybrid];
    
    UINT32 PCmask_hybrid;
    
    
    bool prediction_neural;
    bool prediction_tage;
    NeuralPredictor *_neural;
    TagePredictor *_tage;
    
public:
    // The interface to the four functions below CAN NOT be changed
    PREDICTOR(void);
    ~PREDICTOR(void);
    bool    GetPrediction(UINT32 PC);
    void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
    void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget);
};
#define LOG_BASE 13
#define LOG_GLOBAL 12
#define N_BANKS 4
#define CTR_BITS 3
#define TAG_BITS 11

#define MAX_LENGTH 35     //131
#define MIN_LENGTH 3
struct folded_history {
    unsigned hash;
    int MOD, ORIGIN_LEN, COMPRESSED_LEN;
    
    void create(int origin_len, int compressed_len) {
        hash = 0;
        ORIGIN_LEN = origin_len;
        COMPRESSED_LEN = compressed_len;
        MOD = ORIGIN_LEN % COMPRESSED_LEN;
    }
    
    void update(bitset<MAX_LENGTH> h) {
        hash = (hash << 1) | h[0];
        hash ^= h[ORIGIN_LEN] << MOD;
        hash ^= (hash >> COMPRESSED_LEN);
        hash &= (1 << COMPRESSED_LEN) - 1;
    }
};
#define Neural_BANKS 4
#define Neural_LOG 9
struct neural_entry {
    int  tag, ubit;
};
class NeuralPredictor{
private:
    // # neurals.
    static const int kSize = 512;
    neural_entry neural_table[Neural_BANKS][1 << Neural_LOG];
    folded_history comp_hist_i[Neural_BANKS], comp_hist_t[2][Neural_BANKS];
    bitset<MAX_LENGTH> global_history;
    int path_history;
    int G_INDEX[Neural_BANKS];
    int lens[Neural_BANKS]; 
    int bank, alt_bank;  
    // The size of the global history shift register (implemented as a circular buffer)
    static const int kHistorySize = 63;
    // The theta value used as the threshold for determining weight saturation.
    int kTheta;
    // The number of bits each weight can use.
    int kWeightSize;
    int w_max;
    int w_min;
    
    UINT64 GHR;
    int ty_out[Neural_BANKS];
    int _y_out;
    bool alt_prediction;
    // The bias, i.e., w_0 for each neural.
    int8_t *_bias;
    int8_t *t_bias;
    
    // The array of neurals.
    int8_t *_weights;

    int8_t weight[Neural_BANKS][1 << Neural_LOG][kHistorySize];
    
    inline int get_key(UINT32 pc) {
        return ((pc >> 2) ^ GHR) % kSize;
    }

    inline int sign(int val) {
        return (val > 0) - (val < 0);
    }
    int get_g_index(UINT32 PC, int bank) {
        int index = PC ^
        (PC >> ((Neural_LOG - Neural_BANKS + bank + 1))) ^
        comp_hist_i[bank].hash;
        if (lens[bank] >= 16)
            index ^= mix_func(path_history, 16, bank);
        else
            index ^= mix_func(path_history, lens[bank], bank);
        return index & ((1 << Neural_LOG) - 1);
    }
    int g_tag(UINT32 PC, int bank) {
        int temp_tag = PC ^ comp_hist_t[0][bank].hash ^ (comp_hist_t[1][bank].hash << 1);
        return temp_tag & ((1 << (TAG_BITS - ((bank + (Neural_BANKS & 1)) / 2))) - 1);
    }
    
    int mix_func(int hist, int size, int bank) {
        hist = hist & ((1 << size) - 1);
        int temp_2 = hist >> Neural_LOG;
        temp_2 = ((temp_2 << bank) & ((1 << Neural_LOG) - 1)) + (temp_2 >> (Neural_LOG - bank));
        int temp_1 = hist & ((1 << Neural_LOG) - 1);
        hist = temp_1 ^ temp_2;
        return ((hist << bank) & ((1 << Neural_LOG) - 1)) + (hist >> (Neural_LOG - bank));
    }
    void alloc_new_hist(bool taken, UINT32 PC) {
        int minu = 3, index = 0;
        for (int i = 0; i < bank; i ++) {
            if (neural_table[i][G_INDEX[i]].ubit < minu) {
                minu = neural_table[i][G_INDEX[i]].ubit;
                index = i;
            }
        }
        if (minu > 0) {
            for (int i = 0; i < bank; i ++) {
                neural_table[i][G_INDEX[i]].ubit --;
            }
        }
        else {
            for(int i = 0; i < kHistorySize; i++){
                weight[index][G_INDEX[index]][i] = 0;              
            }
            neural_table[index][G_INDEX[index]].tag = g_tag(PC, index);
            neural_table[index][G_INDEX[index]].ubit = 0;
        }
    }
    void get_index(UINT32 PC){
        for (int i = 0; i < Neural_BANKS; i++){
            G_INDEX[i] = get_g_index(PC, i);
        }
    }
    void lookup(UINT32 PC){
        bank = alt_bank = Neural_BANKS;
        for (int i = 0; i < Neural_BANKS; i++){
            if (neural_table[i][G_INDEX[i]].tag == g_tag(PC, i)){
                bank = i;
                break;
            }            
        }
        for (int i = bank + 1; i < Neural_BANKS; i++){
            if (neural_table[i][G_INDEX[i]].tag == g_tag(PC, i)){
                alt_bank = i;
                break;
            }
        }
    }
    int sum_weights(UINT32 PC, int bank){
        int sum = 0;
        for (int j = 0; j < kHistorySize; ++j){
            int xi = (((GHR >> j) & 0x1) == 1) ? 1 : -1;
            sum += weight[bank][G_INDEX[bank]][j] * xi;
        }
        return sum;
    }
    bool neural_output(UINT32 PC, int bank){
            ty_out[bank] = 0;
            for (int j = 0; j < kHistorySize; ++j){
                int xi = (((GHR >> j) & 0x1) == 1) ? 1 : -1;
                ty_out[bank] += weight[bank][G_INDEX[bank]][j] * xi;
            } 
            return ty_out[bank] >= 0;
    }
    void updat_weights(int bank, bool resolveDir){
        int t = resolveDir ? 1 : -1;
        if (sign(ty_out[bank]) != t || abs(ty_out[bank]) <= kTheta){
            for (int j = 0; j < kHistorySize; ++j){
                int xi = (((GHR >> j) & 0x1) == 1) ? 1 : -1;
                int t_wi= weight[bank][G_INDEX[bank]][j] + xi * t;
                if (t_wi >= w_min && t_wi <= w_max){
                    weight[bank][G_INDEX[bank]][j] = t_wi;
                }
            }
        }
        
    }
    void update_hist(bool taken, UINT32 PC) {
        path_history = (path_history << 1) + (PC & 1);
        path_history &= (1 << 10) - 1;
        global_history <<= 1;
        if (taken)
            global_history = global_history | (bitset<MAX_LENGTH>)1;
        for (int i = 0; i < Neural_BANKS; i ++) {
            comp_hist_t[0][i].update(global_history);
            comp_hist_t[1][i].update(global_history);
            comp_hist_i[i].update(global_history);
        }
    }
    bool get_base_pred(UINT32 PC) {
        int key = get_key(PC);
        int y = _bias[key];
        for (int i = 0; i < kHistorySize; ++i) {
            int xi = (((GHR >> i) & 0x1) == 1) ? 1 : -1;
            y += _weights[key*kHistorySize+i] * xi;
        }   
        _y_out = y;
    //    printf("y = %d\n", y);
        return y >= 0;
    }
    void update_base(UINT32 PC, bool taken) {
    int t = taken ? 1 : -1;
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
       
    }
    
public:
    // The interface to the four functions below CAN NOT be changed
    NeuralPredictor(void);
    ~NeuralPredictor(void);
    // Computes the y value of the neural with the given key.
    bool    GetPrediction(UINT32 PC);
    // Trains the neural associated with PC, given its previously-computed
    // y value as well as t, denoting whether or not the branch was taken.
    void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
    void    UpdateGHR(UINT64 GHR);
};


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

struct base_entry {
    int pred;
    base_entry(): pred(0) {}
};

struct global_entry {
    int ctr, tag, ubit;
    global_entry(): tag(0), ubit(random() & 3) {
        ctr = (random() & ((1 << CTR_BITS) - 1)) - (1 << (CTR_BITS - 1));
    }
};

class TagePredictor{
    
    // The state is defined for Gshare, change for your design
    
private:
    global_entry global_table[N_BANKS][1 << LOG_GLOBAL];
    base_entry base_table[1 << LOG_BASE];
    
    folded_history comp_hist_i[N_BANKS], comp_hist_t[2][N_BANKS];
    bitset<MAX_LENGTH> global_history;
    
    int path_history;
    int G_INDEX[N_BANKS], B_INDEX;
    int lens[N_BANKS];
    int bank, alt_bank;
    
    int pred_store;
    bool alt_pred;
    
    
    int get_b_index(UINT32 PC) {
        return PC & ((1 << LOG_BASE) - 1);
    }
    
    int get_g_index(UINT32 PC, int bank) {
        int index = PC ^
        (PC >> ((LOG_GLOBAL - N_BANKS + bank + 1))) ^
        comp_hist_i[bank].hash;
        if (lens[bank] >= 16)
            index ^= mix_func(path_history, 16, bank);
        else
            index ^= mix_func(path_history, lens[bank], bank);
        return index & ((1 << LOG_GLOBAL) - 1);
    }
    
    int g_tag(UINT32 PC, int bank) {
        int temp_tag = PC ^ comp_hist_t[0][bank].hash ^ (comp_hist_t[1][bank].hash << 1);
        return temp_tag & ((1 << (TAG_BITS - ((bank + (N_BANKS & 1)) / 2))) - 1);
    }
    
    int mix_func(int hist, int size, int bank) {
        hist = hist & ((1 << size) - 1);
        int temp_2 = hist >> LOG_GLOBAL;
        temp_2 = ((temp_2 << bank) & ((1 << LOG_GLOBAL) - 1)) + (temp_2 >> (LOG_GLOBAL - bank));
        int temp_1 = hist & ((1 << LOG_GLOBAL) - 1);
        hist = temp_1 ^ temp_2;
        return ((hist << bank) & ((1 << LOG_GLOBAL) - 1)) + (hist >> (LOG_GLOBAL - bank));
    }
    
    void update_base(UINT32 PC, bool taken) {
        if (taken) {
            if (base_table[B_INDEX].pred < 1)
                base_table[B_INDEX].pred ++;
        }
        else {
            if (base_table[B_INDEX].pred > - 2)
                base_table[B_INDEX].pred --;
        }
    }
    
    void update_ctr(int &ctr, bool taken, int bits) {
        if (taken) {
            if (ctr < ((1 << (bits - 1)) - 1))
                ctr ++;
        }
        else {
            if (ctr > - (1 << (bits - 1)))
                ctr --;
        }
    }
    
    void alloc_new_hist(bool taken, UINT32 PC) {
        int minu = 3, index = 0;
        for (int i = 0; i < bank; i ++) {
            if (global_table[i][G_INDEX[i]].ubit < minu) {
                minu = global_table[i][G_INDEX[i]].ubit;
                index = i;
            }
        }
        if (minu > 0) {
            for (int i = 0; i < bank; i ++) {
                global_table[i][G_INDEX[i]].ubit --;
            }
        }
        else {
            global_table[index][G_INDEX[index]].ctr = taken ? 0 : -1;
            global_table[index][G_INDEX[index]].tag = g_tag(PC, index);
            global_table[index][G_INDEX[index]].ubit = 0;
        }
    }
    
    void update_hist(bool taken, UINT32 PC) {
        path_history = (path_history << 1) + (PC & 1);
        path_history &= (1 << 10) - 1;
        global_history <<= 1;
        if (taken)
            global_history = global_history | (bitset<MAX_LENGTH>)1;
        for (int i = 0; i < N_BANKS; i ++) {
            comp_hist_t[0][i].update(global_history);
            comp_hist_t[1][i].update(global_history);
            comp_hist_i[i].update(global_history);
        }
    }
    
    void calc_index(UINT32 PC) {
        B_INDEX = get_b_index(PC);
        for (int i = 0; i < N_BANKS; i ++) {
            G_INDEX[i] = get_g_index(PC, i);
        }
    }
    
    void linear_search(UINT32 PC) {
        bank = alt_bank = N_BANKS;
        for (int i = 0; i < N_BANKS; i ++) {
            if (global_table[i][G_INDEX[i]].tag == g_tag(PC, i)) {
                bank = i;
                break;
            }
        }
        for (int i = bank + 1; i < N_BANKS; i ++) {
            if (global_table[i][G_INDEX[i]].tag == g_tag(PC, i)) {
                alt_bank = i;
                break;
            }
        }
    }
    
    bool get_base_pred(UINT32 PC) {
        return base_table[B_INDEX].pred >= 0;
    }
    
    
public:
    
    // The interface to the four functions below CAN NOT be changed
    
    TagePredictor(void);
    bool    GetPrediction(UINT32 PC);
    void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
    void    UpdateGHR(UINT64 GHR);

    // Contestants can define their own functions below
    
};

#endif

