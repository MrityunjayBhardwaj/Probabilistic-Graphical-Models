const node_names = ['x1', 'x2', 'x3', 'x4'];
const dims1 = [2, 3, 2, 2];

function createVars(){
    return new Array(node_names.length).fill(0).map(
        (_, i)=>new Variable(node_names[i], dims[i])
    )
}
// pad array just so that reference like x[1] later is easier to read
const x = createVars();

const f3 = new Factor('f3', tf.tensor([0.2, 0.8]).expandDims(1));
const f4 = new Factor('f4', tf.tensor([0.5, 0.5]).expandDims(1));

// first index is x3, second index is x4, third index is x2
// looking at it like: arr[0][0][0]
const f234 = new Factor('f234', tf.tensor([
    [
        [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]
    ], [
        [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]
    ]
]));

// first index is x2
const f12 = new Factor('f12', tf.tensor([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]));


const g = new FactorGraph(x[3], silent=false, debug=false);
g.append('x3', f234);
g.append('f234', x[4]);
g.append('f234', x[2]);
g.append('x2', f12);
g.append('f12', x[1]);
g.append('x3', f3);
g.append('x4', f4);

g.computeMarginals();


g.observe(['x2'], [2]).then(
    ()=>{

        console.log('after observe yes!')
        g.computeMarginals();
        const newMarginals = g.exportMarginals();

        newMarginals.x1.print();
        newMarginals.x2.print();
        newMarginals.x3.print();
        newMarginals.x4.print();
    }
)
