import { NavLink, Route, Routes } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Suggestions from './pages/Suggestions.jsx'
import Stocks from './pages/Stocks.jsx'
import StockDetail from './pages/StockDetail.jsx'
import Runs from './pages/Runs.jsx'
import Contact from './pages/Contact.jsx'

const links = [
  ['/', 'Home', true],
  ['/suggestions', 'Suggestions'],
  ['/stocks', 'Stocks'],
  ['/runs', 'Runs'],
  ['/contact', 'Contact'],
]

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">StockBoiz <span className="v">v3</span></NavLink>
        <nav>
          {links.map(([to, label, end]) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => (isActive ? 'active' : '')}>{label}</NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/suggestions" element={<Suggestions />} />
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="*" element={<div className="card"><h2>Page not found</h2><p>Nothing lives at this address.</p></div>} />
        </Routes>
      </main>
      <footer>Every number here is net of modelled costs and comes from the quantlab pipeline. Nothing on this page is investment advice.</footer>
    </div>
  )
}
