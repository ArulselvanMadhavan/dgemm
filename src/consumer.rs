use dam::context_tools::*;

#[context_macro]
pub struct Consumer<T: DAMType> {
    capacity: u64,
    input: Receiver<T>,
    node_id: usize,
}

impl<T: DAMType> Consumer<T> {
    pub fn new(capacity: u64, input: Receiver<T>, node_id: usize) -> Self {
        let result = Self {
            capacity,
            input,
            node_id,
            context_info: Default::default(),
        };
        result.input.attach_receiver(&result);
        result
    }
}

impl<T> Context for Consumer<T>
where
    T: DAMType,
{
    fn run(&mut self) {
        let mut count: u64 = 0;
        loop {
            match self.input.dequeue(&self.time) {
                Ok(_x) => {
                    // println!(
                    //     "Consumer:{:?}|{:?}|{:?}",
                    //     self.node_id,
                    //     self.time.tick(),
                    //     x.data
                    // );
                    count += 1;
                }
                Err(_) => return,
            }
            if count == self.capacity {
                self.time.incr_cycles(1);
                count = 0;
            }
        }
    }
}
