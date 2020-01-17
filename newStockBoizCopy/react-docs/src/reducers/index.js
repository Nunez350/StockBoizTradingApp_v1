import stockReducer from './stockReducer';
import loggedReducer from './isLogged';
import { combineReducers} from 'redux';

const allReducers = combineReducers({
    stockTrack : stockReducer,
    isLogged : loggedReducer
})

export default allReducers;