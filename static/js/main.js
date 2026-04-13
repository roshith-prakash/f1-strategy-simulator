document.addEventListener('DOMContentLoaded', () => {
    const yearSlider = document.getElementById('year');
    const yearVal = document.getElementById('year-val');
    const runBtn = document.getElementById('run-btn');
    const loader = document.getElementById('loader');
    const welcomeView = document.getElementById('welcome-view');
    const resultsView = document.getElementById('results-view');
    const leaderboardGrid = document.getElementById('leaderboard-grid');
    const lapTableBody = document.querySelector('#lap-table tbody');
    const errorModal = document.getElementById('error-modal');
    const errorMsg = document.getElementById('error-msg');
    const closeBtn = document.querySelector('.close-btn');

    // Update year label on slider input
    yearSlider.addEventListener('input', (e) => {
        yearVal.textContent = e.target.value;
    });

    // Close error modal
    closeBtn.addEventListener('click', () => {
        errorModal.classList.add('hidden');
    });

    // Handle simulation run
    runBtn.addEventListener('click', async () => {
        const config = {
            driver: document.getElementById('driver').value,
            race: document.getElementById('race').value,
            year: yearSlider.value
        };

        // Show loading state
        loader.classList.remove('hidden');
        runBtn.disabled = true;

        try {
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
                welcomeView.classList.add('hidden');
                resultsView.classList.remove('hidden');
            } else {
                showError(data.error || 'Unknown simulation error occurred.');
            }
        } catch (err) {
            showError('Failed to connect to simulation engine.');
            console.error(err);
        } finally {
            loader.classList.add('hidden');
            runBtn.disabled = false;
        }
    });

    function renderResults(data) {
        // Clear previous results
        leaderboardGrid.innerHTML = '';
        lapTableBody.innerHTML = '';
        const actualContainer = document.getElementById('actual-strategy-container');
        actualContainer.innerHTML = '';

        // Render Actual Historical Strategy
        if (data.actual) {
            actualContainer.innerHTML = `
                <span class="label">ACTUAL RACE:</span>
                <span class="value">${data.actual.strategy}</span>
                <span class="time">${data.actual.total_time_str}</span>
            `;
        } else {
            actualContainer.innerHTML = `<span class="no-data-tag">Historical data not available for this session</span>`;
        }

        // Render Leaderboard Cards
        data.leaderboard.forEach((item, index) => {
            const rank = index + 1;
            const isBest = rank === 1;
            
            const card = document.createElement('div');
            card.className = `strategy-card ${isBest ? 'best active' : ''}`;
            card.dataset.index = index;
            
            card.innerHTML = `
                <div class="card-rank">${rank}</div>
                ${isBest ? '<span class="card-label">RECOMMENDED STRATEGY</span>' : ''}
                <div class="card-compounds">${item.strategy}</div>
                <div class="card-stats">
                    <div class="stat">
                        <span class="stat-val">${item.total_time_str}</span>
                        <span class="stat-label">Total Race Time</span>
                    </div>
                    <div class="stat">
                        <span class="stat-val">${item.n_pitstops}</span>
                        <span class="stat-label">Pit Stops</span>
                    </div>
                </div>
                <div class="card-footer">CLICK TO VIEW TELEMETRY</div>
            `;

            card.addEventListener('click', () => {
                // Update active state
                document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                
                // Update Table
                renderLapTable(item.laps);
                
                // Scroll to table smoothly on mobile/small screens
                if (window.innerWidth < 1024) {
                    document.querySelector('.telemetry-section').scrollIntoView({ behavior: 'smooth' });
                }
            });

            leaderboardGrid.appendChild(card);
        });

        // Initial render of the best strategy table
        if (data.leaderboard.length > 0) {
            renderLapTable(data.leaderboard[0].laps);
        }
    }

    function renderLapTable(laps) {
        lapTableBody.innerHTML = '';
        laps.forEach(lap => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${lap.Lap}</td>
                <td style="font-weight: 600;">${lap.LapTime}s</td>
                <td><span class="compound-badge comp-${lap.Compound}">${lap.Compound}</span></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem">
                        <div style="flex: 1; height: 4px; background: #2d333b; border-radius: 2px">
                            <div style="width: ${lap.TyreDeg}%; height: 100%; background: ${getDegColor(lap.TyreDeg)}; border-radius: 2px"></div>
                        </div>
                        <span style="font-size: 0.75rem; color: var(--text-dim); min-width: 40px">${lap.TyreDeg}%</span>
                    </div>
                </td>
                <td style="color: var(--text-dim)">#${lap.Stint}</td>
            `;
            lapTableBody.appendChild(row);
        });
    }

    function getDegColor(deg) {
        if (deg < 50) return '#00d2be'; // Success/Green
        if (deg < 80) return '#ffd000'; // Warning/Yellow
        return '#ff0000'; // Error/Red
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorModal.classList.remove('hidden');
    }
});
