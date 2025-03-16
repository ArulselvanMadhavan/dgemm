use dam::context_tools::*;

#[context_macro]
pub struct Actfn<T: Clone> {
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
    func: fn(T) -> T,
}

impl<T: DAMType> Actfn<T> {
    pub fn new(
        input: Receiver<T>,
        output: Sender<T>,
        initiation_interval: u64,
        func: fn(T) -> T,
    ) -> Self {
        let result = Self {
            input,
            output,
            initiation_interval,
            func,
            context_info: Default::default(),
        };
        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

impl<T: DAMType> Context for Actfn<T> {
    fn run(&mut self) {
        loop {
            match self.input.dequeue(&self.time) {
                Ok(data) => self
                    .output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(data.time + 1, (self.func)(data.data)),
                    )
                    .unwrap(),
                Err(_) => return,
            }
            self.time.incr_cycles(self.initiation_interval)
        }
    }
}
