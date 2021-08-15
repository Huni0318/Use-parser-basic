94
            'accuracy': float(accuracy),
95
        }
96
​
97
    @staticmethod
98
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
99
        # Attaching would be repaeted for serveral metrics.
100
        # Thus, we can reduce the repeated codes by using this function.
101
        def attach_running_average(engine, metric_name):
102
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
103
                engine,
104
                metric_name,
105
            )
106
​
107
        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']
108
​
109
        for metric_name in training_metric_names:
110
            attach_running_average(train_engine, metric_name)
111
​
112
        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
113
        # Without ignite, you can use tqdm to implement progress bar.
114
        if verbose >= VERBOSE_BATCH_WISE:
115
            pbar = ProgressBar(bar_format=None, ncols=120)
116
            pbar.attach(train_engine, training_metric_names)
117
​
118
        # If the verbosity is set, statistics would be shown after each epoch.
119
        if verbose >= VERBOSE_EPOCH_WISE:
120
            @train_engine.on(Events.EPOCH_COMPLETED)
121
            def print_train_logs(engine):
122
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
123
                    engine.state.epoch,
124
                    engine.state.metrics['|param|'],
125
                    engine.state.metrics['|g_param|'],
126
                    engine.state.metrics['loss'],
127
                    engine.state.metrics['accuracy'],
128
                ))
129
​
130
        validation_metric_names = ['loss', 'accuracy']
131
        
132
        for metric_name in validation_metric_names:
133
            attach_running_average(validation_engine, metric_name)
134
​
135
        # Do same things for validation engine.
136
        if verbose >= VERBOSE_BATCH_WISE:
137
            pbar = ProgressBar(bar_format=None, ncols=120)
138
            pbar.attach(validation_engine, validation_metric_names)
139
​
140
        if verbose >= VERBOSE_EPOCH_WISE:
141
            @validation_engine.on(Events.EPOCH_COMPLETED)
142
            def print_valid_logs(engine):
143
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
144
                    engine.state.metrics['loss'],
145
                    engine.state.metrics['accuracy'],
146
                    engine.best_loss,
147
                ))
148
​
149
    @staticmethod
150
    def check_best(engine):
151
        loss = float(engine.state.metrics['loss'])
152
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
153
            engine.best_loss = loss  # Update lowest validation loss.
154
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.
155
​
156
    @staticmethod
157
    def save_model(engine, train_engine, config, **kwargs):
158
        torch.save(
159
            {
160
                'model': engine.best_model,
161
                'config': config,
162
                **kwargs
163
            }, config.model_fn
164
        )
165
​
166
​
167
class Trainer():
168
​
169
    def __init__(self, config):
170
        self.config = config
171
​
172
    def train(
173
        self,
174
        model, crit, optimizer,
175
        train_loader, valid_loader
176
    ):
177
        train_engine = MyEngine(
178
            MyEngine.train,
179
            model, crit, optimizer, self.config
180
        )
181
        validation_engine = MyEngine(
182
            MyEngine.validate,
183
            model, crit, optimizer, self.config
184
        )
185
​
186
        MyEngine.attach(
187
            train_engine,
188
            validation_engine,
189
            verbose=self.config.verbose
190
        )
191
​
192
        def run_validation(engine, validation_engine, valid_loader):
193
            validation_engine.run(valid_loader, max_epochs=1)
194
​
195
        train_engine.add_event_handler(
196
            Events.EPOCH_COMPLETED, # event
197
            run_validation, # function
198
            validation_engine, valid_loader, # arguments
199
        )
200
        validation_engine.add_event_handler(
201
            Events.EPOCH_COMPLETED, # event
202
            MyEngine.check_best, # function
203
        )
204
        validation_engine.add_event_handler(
205
            Events.EPOCH_COMPLETED, # event
206
            MyEngine.save_model, # function
207
            train_engine, self.config, # arguments
208
        )
209
​
210
        train_engine.run(
211
            train_loader,
212
            max_epochs=self.config.n_epochs,
213
        )
214
​
215
        model.load_state_dict(validation_engine.best_model)
216
​
217
        return model
