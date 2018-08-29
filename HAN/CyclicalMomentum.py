from keras.callbacks import *

class CyclicMT(Callback):
    def __init__(self, base_mt=0.85, max_mt=0.95, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicMT, self).__init__()

        self.base_mt = base_mt
        self.max_mt = max_mt
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.cmt_iterations = 0.
        self.trn_iterations = 0.
        #self.history = {}

        self._reset()

    def _reset(self, new_base_mt=None, new_max_mt=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_mt != None:
            self.base_mt = new_base_mt
        if new_max_mt != None:
            self.max_mt = new_max_mt
        if new_step_size != None:
            self.step_size = new_step_size
        self.cmt_iterations = 0.
        
    def cmt(self):
        cycle = np.floor(1+self.cmt_iterations/(2*self.step_size))
        x = np.abs(self.cmt_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_mt + (-self.max_mt+self.base_mt)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_mt + (-self.max_mt+self.base_mt)*np.maximum(0, (1-x))*self.scale_fn(self.cmt_iterations)
        
    def on_train_begin(self, logs={}):
        #logs = logs or {}

        if self.cmt_iterations == 0:
            K.set_value(self.model.optimizer.momentum, self.base_mt)
        else:
            K.set_value(self.model.optimizer.momentum, self.cmt())        
            
    def on_batch_end(self, epoch, logs=None):
        
        #logs = logs or {}
        self.trn_iterations += 1
        self.cmt_iterations += 1

        #self.history.setdefault('mt', []).append(K.get_value(self.model.optimizer.momentum))
        #self.history.setdefault('iterations', []).append(self.trn_iterations)

        #for k, v in logs.items():
        #    self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.momentum, self.cmt())
