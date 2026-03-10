import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FactorOracle:
    def __init__(self):
        # symbols of the oracle
        self.symbols = [] 
        # for each state, sigma stores a list of transitions for each symbol
        self.sigma = [] 
        # suffix links for each state
        self.S = [] 
        
    def train(self, sequence):
        w = sequence
        self.S.append(None)
        for i in range(1,len(w)+1):
            i_word = i-1
            if w[i_word] not in self.symbols: # add new symbol to dictionary
                self.symbols.append(w[i_word])
                for ss in self.sigma: # add new word space to past sigmas
                    ss.append(None)
            for j in range(len(self.symbols)): # find index of current symbol
                if self.symbols[j] == w[i_word]:
                    symbolIdx = j
            # initialize empty states
            self.S.append(None)
            self.sigma.append([None for _ in range(len(self.symbols))])
            self.sigma[i-1][symbolIdx] = i
            k = self.S[i-1]
            while k is not None and self.sigma[k][symbolIdx] is None:
                self.sigma[k][symbolIdx] = i
                k = self.S[k]
            if k is not None:
                self.S[i] = self.sigma[k][symbolIdx]
            else:
                self.S[i] = 0
        self.sigma.append([None for _ in range(len(self.symbols))])

    def predict(self, sequence=[], num_predictions=1, p=1):
        state = 0
        v = sequence
        # look for factor matching input sequence
        if all(char in sequence for char in self.symbols):
            foundMatch = False
            while not foundMatch:
                for i in range(len(v)):
                    # find index of current symbol
                    for j in range(len(self.symbols)): 
                        if self.symbols[j] == v[i]:
                            symbolIdx = j
                    if state is not None:
                        state = self.sigma[state][symbolIdx]
                        foundMatch = True
                    else:
                        foundMatch = False
                        break
                if v != []:
                    v = v[1:]
                else:
                    break
        i = state if state is not None else 0
        # generate predicted sequence
        preds = []
        seq_len = num_predictions
        for n in range(seq_len):
            q = 1 if self.S[i] == None else p
            if random.random() < q and i < len(self.S)-1:
                idx = self.sigma[i].index(i+1)
                preds.append(self.symbols[idx])
                i += 1
            else:
                sig = self.sigma[self.S[i]]
                not_none_idxs = [j for j in range(len(sig)) if sig[j] is not None]
                idx = random.choice(not_none_idxs)
                preds.append(self.symbols[idx])
                i = self.sigma[self.S[i]][idx]
        return preds
    
    def visualize(self):
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")
        suffix_style = "Simple, tail_width=0.3, head_width=4, head_length=8"
        kw_suf = dict(arrowstyle=suffix_style, color="k")
        N_states = len(self.S)
        # start a new figure
        fig, ax = plt.subplots(1, figsize=(N_states+1, 6))
        longest_delta = 1
        for s in range(len(self.S)):
            circle = plt.Circle((0.5+s, 0.5), 0.2, color='black', fill=False)
            ax.add_patch(circle)
            label = ax.annotate(s, xy=(0.5+s, 0.5), fontsize=15, verticalalignment="center", horizontalalignment="center")
            for i in range(len(self.sigma[s])):
                delta = self.sigma[s][i]
                if delta == s+1:
                    a = patches.FancyArrowPatch((0.5+s+0.2, 0.5), (0.5+(delta)-0.2, 0.5), **kw)
                    plt.gca().add_patch(a)
                    label = ax.annotate(self.symbols[i], xy=((delta-s)+s, 0.6), fontsize=15, verticalalignment="center", horizontalalignment="center")
                elif delta is not None:
                    a = patches.FancyArrowPatch((0.5+s, 0.7), (0.5+(delta), 0.7), connectionstyle="arc3,rad=-.5", **kw)
                    plt.gca().add_patch(a)
                    label = ax.annotate(self.symbols[i], xy=((delta-s)/2+s+0.5, 0.9+(delta-s)/4), fontsize=15, verticalalignment="center", horizontalalignment="center")
                    if delta-s > longest_delta:
                        longest_delta = delta-s
            if self.S[s] is not None:
                a = patches.FancyArrowPatch((0.5+s, 0.3), (0.5+self.S[s], 0.3), connectionstyle="arc3,rad=-.5", **kw_suf, alpha=0.2)
                plt.gca().add_patch(a)
        ax.set_xlim(0, N_states)
        ax.set_ylim(-longest_delta/3+0.5, longest_delta/3+0.5)
        ax.set_aspect('equal', adjustable='box') # Ensures the circle isn't distorted into an ellipse
        fig.tight_layout()
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    w = "CACIOCAVALLO"
    FO = FactorOracle()
    FO.train(w)
    FO.visualize()
    ff = FO.predict([],10,0.8)
    print(f'generated sequence: {ff}')
