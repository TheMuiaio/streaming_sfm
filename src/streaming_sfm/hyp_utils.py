import sys
import Levenshtein

class ABSHypothesisBuffer:
    def __init__(self, debug=False):
        self.buffer = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.debug = debug

    def insert(self, new, offset=0):
        return NotImplementedError
    
    def flush(self):
        return NotImplementedError
    
    def complete(self):
        return self.buffer
    
    @staticmethod
    def reset(self):
        self.buffer = []
        self.commited_in_buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None


class LCPHypothesisBuffer(ABSHypothesisBuffer):
    def __init__(self, uncased=True, debug=True):
        super(LCPHypothesisBuffer, self).__init__(debug=debug)
        self.commited_in_buffer = [] # Paraules del buffer que ja s'han consolidat
        self.new = [] # Buffer de paraules noves a afegir al buffer

        self.uncased = uncased

    def insert(self, new, offset=0):
        """
        Compara self.commited_in_buffer i new. Afegeix al buffer NOMÉS les
        paraules de new que extenen la llista self.commited_in_buffer.
        La nova cua es guarda en self.new
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> insert] Recieved hypothesis: {new}')
        new = [(a+offset, b+offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time-0.1]
        if self.debug:
            print(f'[LCPHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(min(c_len, n_len), 5) + 1): # Posem 5 com a màxim. Depén molt del tamany de chunk. Es podria especificar com a argument.
                        c = ' '.join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1]) # Darreres $i$ paraules de self.commited_in_buffer
                        tail = ' '.join(self.new[j-1][2] for j in range(1, i+1)) # Primeres $i$ paraules de self.new

                        if c == tail: # Si ambdues són iguals, podem eliminar les primeres $i$ paraules de self.new (ja estàn consolidades)
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = ' '.join(words)
                            if self.debug:
                                print(f'[LCPHypothesisBuffer -> insert] Removing last {i} words: {words_msg}')
                            break

    def flush(self):
        if self.uncased:
            return self.flush_uncased()
        else:
            return self.flush_cased()

    def flush_uncased(self):
        """
        Torna el chunk consolidat usant l'LCP de les darreres 2 iteracions
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Buffer n-1 is {self.buffer}')
            print(f'[LCPHypothesisBuffer -> flush] Buffer n is {self.new}')
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            #if nt in '.,!?-_':
            #    commit.append((na, nb, nt))
            #    self.new.pop(0)
            #    if self.buffer[0][2] in '._,!?-_':
            #        self.buffer.pop(0)
            if nt.lower() == self.buffer[0][2].lower():
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []

        if len(self.commited_in_buffer) and self.commited_in_buffer[-1][2] in punct:
            if len(commit) and commit[0][2] in punct:
                commit = commit[1:]

        if len(commit) < 1:
            return commit

        self.commited_in_buffer.extend(commit)
        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Committing {commit}')
        return commit

    def flush_cased(self):
        """
        Torna el chunk consolidat usant l'LCP de les darreres 2 iteracions
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Buffer n-1 is {self.buffer}')
            print(f'[LCPHypothesisBuffer -> flush] Buffer n is {self.new}')
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Committing {commit}')
        return commit

    def pop_commited(self, time_limit):
        """
        Neteja self.commited_in_buffer fins a un instant de temps donat 'time_limit'
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_limit:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """
        Torna les paraules restants del buffer que encara no han consolidat
        """
        return self.buffer


class LACPHypothesisBuffer(ABSHypothesisBuffer):
    def __init__(self, threshold: int = 2, uncased=True, debug=True):
        super(LACPHypothesisBuffer, self).__init__(debug=debug)
        self.commited_in_buffer = [] # Paraules del buffer que ja s'han consolidat
        self.new = [] # Buffer de paraules noves a afegir al buffer

        self.uncased = uncased
        self.threshold = threshold

    def insert(self, new, offset=0):
        """
        Compara self.commited_in_buffer i new. Afegeix al buffer NOMÉS les
        paraules de new que extenen la llista self.commited_in_buffer.
        La nova cua es guarda en self.new
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> insert] Recieved hypothesis: {new}')
        new = [(a+offset, b+offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time-0.1]
        if self.debug:
            print(f'[LCPHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(min(c_len, n_len), 5) + 1): # Posem 5 com a màxim. Depén molt del tamany de chunk. Es podria especificar com a argument.
                        c = ' '.join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1]) # Darreres $i$ paraules de self.commited_in_buffer
                        tail = ' '.join(self.new[j-1][2] for j in range(1, i+1)) # Primeres $i$ paraules de self.new

                        if c == tail: # Si ambdues són iguals, podem eliminar les primeres $i$ paraules de self.new (ja estàn consolidades)
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = ' '.join(words)
                            if self.debug:
                                print(f'[LCPHypothesisBuffer -> insert] Removing last {i} words: {words_msg}')
                            break

    def flush(self):
        if self.uncased:
            return self.flush_uncased()
        else:
            return self.flush_cased()

    def flush_uncased(self):
        """
        Torna el chunk consolidat usant l'LCP de les darreres 2 iteracions
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Buffer n-1 is {self.buffer}')
            print(f'[LCPHypothesisBuffer -> flush] Buffer n is {self.new}')
        punct = [',', '.', '!', '?']
        commit = []
        past = []
        present = []
        i = 0
        while i in range(0, min(len(self.new), len(self.buffer))):
            past.append(self.buffer[i])
            present.append(self.new[i])

            past_s = ' '.join([t for _, _, t in past]).lower()
            present_s = ' '.join([t for _, _, t in present]).lower()

            if Levenshtein.distance(past_s, present_s) > self.threshold:
                break

            commit.append(self.new[i])
            i += 1

        self.buffer = self.new[i:]
        self.new = []

        if len(self.commited_in_buffer) and self.commited_in_buffer[-1][2] in punct:
            if len(commit) and commit[0][2] in punct:
                commit = commit[1:]

        if len(commit) < 1:
            return commit

        self.last_commited_word = commit[-1][2]
        self.last_commited_time = commit[-1][1]

        self.commited_in_buffer.extend(commit)

        if self.debug:
            print(f'[LACPHypothesisBuffer -> flush] Committing {commit}')
        return commit

    def flush_cased(self):
        """
        Torna el chunk consolidat usant l'LCP de les darreres 2 iteracions
        """

        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Buffer n-1 is {self.buffer}')
            print(f'[LCPHypothesisBuffer -> flush] Buffer n is {self.new}')
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        if self.debug:
            print(f'[LCPHypothesisBuffer -> flush] Committing {commit}')
        return commit

    def pop_commited(self, time_limit):
        """
        Neteja self.commited_in_buffer fins a un instant de temps donat 'time_limit'
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_limit:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """
        Torna les paraules restants del buffer que encara no han consolidat
        """
        return self.buffer


class WaitKHypothesisBuffer(ABSHypothesisBuffer):
    def __init__(
        self,
        K: int = 1,
        features_per_second: int = 12,
        subsampling_factor: int = 4,
        debug=False,
    ):
        super(WaitKHypothesisBuffer, self).__init__(debug=debug)
        self.new = []
        self.commited_in_buffer = []

        # Modificar la K per a que:
        #   - Siga divisible pel numero de frames
        #   - Represente directament el número de frames de les timestamps que representa
        self.K = int(K * features_per_second / subsampling_factor)
        if self.debug:
            print(f'[WaitKHypothesisBuffer -> __init__] K initially set to {K} seconds has changed to {self.K} encoder frames')
    
    def insert(self, new, offset=0):
        """
        Compara self.commited_in_buffer i new. Afegeix al buffer NOMÉS les
        paraules de new que extenen la llista self.commited_in_buffer.
        La nova cua es guarda en self.new
        """

        if self.debug:
            print(f'[WaitKHypothesisBuffer -> insert] Recieved hypothesis: {new}')
        new = [(a+offset, b+offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time-0.1]
        if self.debug:
            print(f'[WaitKHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(min(c_len, n_len), 5) + 1): # Posem 5 com a màxim. Depén molt del tamany de chunk. Es podria especificar com a argument.
                        c = ' '.join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1]) # Darreres $i$ paraules de self.commited_in_buffer
                        tail = ' '.join(self.new[j-1][2] for j in range(1, i+1)) # Primeres $i$ paraules de self.new

                        if c == tail: # Si ambdues són iguals, podem eliminar les primeres $i$ paraules de self.new (ja estàn consolidades)
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = ' '.join(words)
                            if self.debug:
                                print(f'[WaitKHypothesisBuffer -> insert] Removing last {i} words: {words_msg}')
                            break

    def flush(self, last_instant):
        """
        Torna el chunk consolidat usant Wait-K on K són segons
        """

        last_valid_instant = last_instant - self.K

        if self.debug:
            print(f'[WaitKHypothesisBuffer -> flush] Last valid emission instant is {last_valid_instant}')
            print(f'[WaitKHypothesisBuffer -> flush] Commited buffer is {self.buffer}')
            print(f'[WaitKHypothesisBuffer -> flush] New buffer is {self.new}')
        commit = []


        if last_valid_instant > 0:
            for i in self.new:
                na, nb, nt = i
                if na > last_valid_instant:
                    break

                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb

            self.buffer.extend(commit)
            self.new = self.new[len(commit):]
        else:
            commit = []

        if self.debug:
            print(f'[WaitKHypothesisBuffer -> flush] Committing {commit}')
        return commit

    def pop_commited(self):
        return self.buffer
    
    def complete(self):
        return self.new

class HoldNHypothesisBuffer(ABSHypothesisBuffer):
    def __init__(self, N: int = 3, debug=False):
        super(HoldNHypothesisBuffer, self).__init__(debug=debug)
        self.new = []
        self.commited_in_buffer = []

        self.N = N
    
    def insert(self, new, offset=0):
        """
        Compara self.commited_in_buffer i new. Afegeix al buffer NOMÉS les
        paraules de new que extenen la llista self.commited_in_buffer.
        La nova cua es guarda en self.new
        """

        new = [(a+offset, b+offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time-0.1]
        if self.debug:
            print(f'[HoldNHypothesisBuffer -> insert] Recieved hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.buffer:
                    
                    c_len = len(self.buffer)
                    n_len = len(self.new)

                    for i in range(1, min(min(c_len, n_len), 10) + 1): # Posem 5 com a màxim. Depén molt del tamany de chunk. Es podria especificar com a argument.
                        c = ' '.join([self.buffer[-j][2] for j in range(1, i+1)][::-1]) # Darreres $i$ paraules de self.buffer
                        tail = ' '.join(self.new[j-1][2] for j in range(1, i+1)) # Primeres $i$ paraules de self.new

                        if c == tail: # Si ambdues són iguals, podem eliminar les primeres $i$ paraules de self.new (ja estàn consolidades)
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = ' '.join(words)
                            if self.debug:
                                print(f'[HoldNHypothesisBuffer -> insert] Removing last {i} words: {words_msg}')
                            break

    def flush(self):
        """
            Torna el chunk consolidat usant Hold-N
        """

        if self.debug:
            print(f'[HoldNHypothesisBuffer -> flush] New buffer is: {self.new}')
            print(f'[HoldNHypothesisBuffer -> flush] Commited buffer is: {self.buffer}')
        if len(self.new) > self.N:
            commit = self.new[:-self.N]
            self.new = self.new[-self.N:]
            self.buffer.extend(commit)
            self.last_commited_time = commit[-1][1]
            self.last_commited_word = commit[-1][2]
        else:
            commit = []

        if self.debug:
            print(f'[HoldNHypothesisBuffer -> flush] Committing {commit}')
        return commit
    
    def pop_commited(self, time_limit):
        """
            Neteja self.buffer fins a un instant de temps donat 'time_limit'
        """
        while self.buffer and self.buffer[0][1] <= time_limit:
            self.buffer.pop(0)
    
    def complete(self):
        return self.new
