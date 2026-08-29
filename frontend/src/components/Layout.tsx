import { Link, Outlet, useLocation } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/cases', label: 'Recovery Cases' },
  { path: '/portfolio', label: 'Portfolio Optimizer' },
];

export default function Layout() {
  const location = useLocation();
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold text-emerald-400 tracking-tight">
            RecoverAI
          </Link>
          <nav className="flex gap-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-1.5 rounded text-sm font-medium transition ${
                  (item.path === '/' ? location.pathname === '/' : location.pathname.startsWith(item.path))
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded font-medium">
            DEMO MODE
          </span>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
