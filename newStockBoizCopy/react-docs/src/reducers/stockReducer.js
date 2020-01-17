const stockReducer = (state = 0,action) => {
    switch(action.type){
        case 'STOCKTRACK':
            //return state 
            return state+1;
        default:
            return state ;
    }
}

export default stockReducer;