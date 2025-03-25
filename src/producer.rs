use dam::context_tools::*;

#[context_macro]
pub struct Producer<T: DAMType, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    iterator: Option<FType>,
    output: Sender<T>,
    node_id: usize,
    init_delay: usize,
}

impl<T: DAMType, IType, FType> Producer<T, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    pub fn new(iterator: FType, output: Sender<T>, node_id: usize, init_delay: usize) -> Self {
        let result = Self {
            iterator: Some(iterator),
            output,
            node_id,
            init_delay,
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
            self.time.incr_cycles(self.init_delay as u64);
            let current_time = self.time.tick();
            for val in (func)() {
                // println!("Prod-{:?}|{:?}", self.node_id, val);
                self.output
                    .enqueue(&self.time, ChannelElement::new(current_time + 1, val))
                    .unwrap();
                self.time.incr_cycles(1);
            }
        } else {
            panic!("Link - No iterator available.{:?}", self.node_id);
        }
    }
}
