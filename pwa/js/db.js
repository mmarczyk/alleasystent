/* IndexedDB wrapper — simple key/value + workout log store */
const DB = (() => {
  const NAME = 'milcal';
  const VER  = 1;
  let _db = null;

  function open() {
    return new Promise((resolve, reject) => {
      if (_db) return resolve(_db);
      const req = indexedDB.open(NAME, VER);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('kv'))
          db.createObjectStore('kv');
        if (!db.objectStoreNames.contains('workouts')) {
          const ws = db.createObjectStore('workouts', { keyPath: 'id', autoIncrement: true });
          ws.createIndex('date', 'date');
        }
      };
      req.onsuccess = e => { _db = e.target.result; resolve(_db); };
      req.onerror   = () => reject(req.error);
    });
  }

  async function get(key) {
    const db = await open();
    return new Promise((res, rej) => {
      const tx = db.transaction('kv', 'readonly');
      const r  = tx.objectStore('kv').get(key);
      r.onsuccess = () => res(r.result ?? null);
      r.onerror   = () => rej(r.error);
    });
  }

  async function set(key, value) {
    const db = await open();
    return new Promise((res, rej) => {
      const tx = db.transaction('kv', 'readwrite');
      tx.objectStore('kv').put(value, key);
      tx.oncomplete = () => res();
      tx.onerror    = () => rej(tx.error);
    });
  }

  async function saveWorkout(workout) {
    const db = await open();
    return new Promise((res, rej) => {
      const tx = db.transaction('workouts', 'readwrite');
      const r  = tx.objectStore('workouts').add(workout);
      r.onsuccess = () => res(r.result);
      r.onerror   = () => rej(r.error);
    });
  }

  async function getAllWorkouts() {
    const db = await open();
    return new Promise((res, rej) => {
      const tx = db.transaction('workouts', 'readonly');
      const r  = tx.objectStore('workouts').getAll();
      r.onsuccess = () => res(r.result);
      r.onerror   = () => rej(r.error);
    });
  }

  async function clearAll() {
    const db = await open();
    return new Promise((res, rej) => {
      const tx = db.transaction(['kv', 'workouts'], 'readwrite');
      tx.objectStore('kv').clear();
      tx.objectStore('workouts').clear();
      tx.oncomplete = () => res();
      tx.onerror    = () => rej(tx.error);
    });
  }

  return { get, set, saveWorkout, getAllWorkouts, clearAll };
})();
