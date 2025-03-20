use dam::context_tools::*;

#[context_macro]
pub struct Producer<T: DAMType, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    iterator: Option<FType>,
    output: Sender<T>,
}

impl<T: DAMType, IType, FType> Producer<T, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    pub fn new(iterator: FType, output: Sender<T>) -> Self {
        let result = Self {
            iterator: Some(iterator),
            output,
            context_info: Default::default(),
        };
        result.output.attach_sender(&result);
        result
    }
}

impl<T, IType, FType> Context for Producer<T, IType, FType>
where
    T: DAMType,
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    fn run(&mut self) {
        if let Some(func) = self.iterator.take() {
            // let mut count: u64 = 0;
            // let mut latency: u64 = 0;
            let current_time = self.time.tick();
            for val in (func)() {
                // latency = (count / self.capacity) + 1;
                dbg!("Len:{:?}", &val);
                self.output
                    .enqueue(&self.time, ChannelElement::new(current_time + 1, val))
                    .unwrap();
                // count += 1;
                self.time.incr_cycles(1);
            }
            // self.time.incr_cycles(latency);
        } else {
            panic!("Link - No iterator available");
        }
    }
}
