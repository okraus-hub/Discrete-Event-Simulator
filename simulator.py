import heapq
import csv
import matplotlib.pyplot as plt
from typing import List

class Event:
    ENQUEUE = 0
    TRANSMIT = 1
    PROPAGATE = 2
    RECEIVE = 3
    cnt = 0

    def __init__(self, event_type: int, target_node, packet=None, time=None):
        self.target_node = target_node
        self.event_type = event_type
        self.time = time
        self.packet = packet
        self.event_id = Event.cnt
        Event.cnt += 1

    def type_to_str(self):
        return {
            self.ENQUEUE: 'ENQUEUE',
            self.TRANSMIT: 'TRANSMIT',
            self.PROPAGATE: 'PROPAGATE',
            self.RECEIVE: 'RECEIVE'
        }[self.event_type]

    def __str__(self):
        return f'{self.time:4d} {self.type_to_str():12s} {self.target_node.node_id} pkt={self.packet}'
    
    def __lt__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.event_id) < (other.time, other.event_id)
    
    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.event_id) == (other.time, other.event_id)

class EventWrapper:
    def __init__(self, time, event):
        self.time = time
        self.event = event
    
    def __lt__(self, other):
        return self.time < other.time

class Packet:
    cnt = 0
    def __init__(self, source, destination):
        self.packet_id = Packet.cnt
        Packet.cnt += 1
        self.source = source
        self.destination = destination
        self.queue_time = None
        self.transmit_time_src = None
        self.receive_time_switch = None
        self.transmit_time_switch = None
        self.receive_time_dest = None

    def __str__(self):
        return f'P[{self.packet_id},{self.source.node_id}->{self.destination.node_id}]'

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.routes = {}
        self.queue = []
        self.is_transmitting = False

    def add_route(self, destination, next_hop):
        self.routes[destination] = next_hop

    def get_next_hop(self, destination):
        return self.routes.get(destination)

class Host(Node):
    def __str__(self):
        return f'{self.node_id:2s} queue={[p.packet_id for p in self.queue]}'

class Switch(Node):
    def __init__(self, node_id, processing_delay=0):
        super().__init__(node_id)
        self.processing_delay = processing_delay

    def __str__(self):
        return f'{self.node_id:2s} queue={[p.packet_id for p in self.queue]}'

class Simulator:
    def __init__(self, max_ticks, transmission_delay=1, propagation_delay=2):
        self.max_ticks = max_ticks
        self.transmission_delay = transmission_delay
        self.propagation_delay = propagation_delay
        self.event_queue = []
        self.time = 0
        self.nodes = {}
        self.packet_records = []
        self.is_single_link = False

    def new_node(self, node_id: str):
        node = Host(node_id)
        self.nodes[node_id] = node
        return node

    def new_switch(self, node_id, processing_delay):
        switch = Switch(node_id, processing_delay)
        self.nodes[node_id] = switch
        return switch

    def schedule_event_after(self, event_type, time, target_node, packet=None):
        event = Event(event_type, target_node, packet, time)
        # Use EventWrapper to handle the comparison
        heapq.heappush(self.event_queue, EventWrapper(time, event))

    def handle_enqueue(self, event):
        node = event.target_node
        packet = event.packet
        packet.queue_time = self.time
        
        if not node.is_transmitting:
            node.is_transmitting = True
            self.schedule_event_after(Event.TRANSMIT, self.time + self.transmission_delay, 
                                    node, packet)
        else:
            node.queue.append(packet)

    def handle_transmit(self, event):
        node = event.target_node
        packet = event.packet
        
        if isinstance(node, Host):
            packet.transmit_time_src = self.time
        else:
            packet.transmit_time_switch = self.time

        next_hop = node.get_next_hop(packet.destination)
        if next_hop:
            self.schedule_event_after(Event.PROPAGATE, self.time + self.propagation_delay,
                                    next_hop, packet)
        
        if node.queue:
            next_packet = node.queue.pop(0)
            self.schedule_event_after(Event.TRANSMIT, self.time + self.transmission_delay,
                                    node, next_packet)
        else:
            node.is_transmitting = False

    def handle_propagate(self, event):
        node = event.target_node
        packet = event.packet
        self.schedule_event_after(Event.RECEIVE, self.time + self.propagation_delay,
                                node, packet)

    def handle_receive(self, event):
        node = event.target_node
        packet = event.packet

        if node == packet.destination:
            packet.receive_time_dest = self.time
            self.record_packet(packet)
        else:
            if isinstance(node, Switch):
                packet.receive_time_switch = self.time
            self.handle_enqueue(event)

    def handle_event(self, event):
        handlers = {
            Event.ENQUEUE: self.handle_enqueue,
            Event.TRANSMIT: self.handle_transmit,
            Event.PROPAGATE: self.handle_propagate,
            Event.RECEIVE: self.handle_receive
        }
        handlers[event.event_type](event)

    def run(self):
        print('Starting simulation')
        while self.event_queue:
            event_wrapper = heapq.heappop(self.event_queue)
            self.time = event_wrapper.time
            if self.time > self.max_ticks:
                break
            self.handle_event(event_wrapper.event)

    def record_packet(self, packet):
        if self.is_single_link:
            self.packet_records.append([
                packet.packet_id,
                packet.queue_time,
                packet.transmit_time_src,
                packet.transmit_time_src + self.propagation_delay,  # Propagate @A
                packet.receive_time_dest
            ])
        else:
            # Calculate delays
            end_to_end_delay = packet.receive_time_dest - packet.queue_time
            queuing_delay = packet.transmit_time_src - packet.queue_time
            # For switch queuing, add any time spent in switch queue
            if packet.transmit_time_switch and packet.receive_time_switch:
                queuing_delay += packet.transmit_time_switch - packet.receive_time_switch
            
            # Total propagation delay (2 links)
            total_propagation_delay = 2 * self.propagation_delay
            
            # Total transmission delay (2 transmissions)
            total_transmission_delay = 2 * self.transmission_delay

            self.packet_records.append([
                packet.packet_id,
                packet.source.node_id,
                packet.queue_time,
                packet.transmit_time_src,
                packet.receive_time_switch,
                packet.transmit_time_switch,
                packet.receive_time_dest,
                end_to_end_delay,
                queuing_delay,
                total_propagation_delay,
                total_transmission_delay
            ])
            

    def write_results(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            if self.is_single_link:
                writer.writerow([
                    "Seq num", "Queue @A", "Transmit @A", 
                    "Propagate @A", "Receive @B"
                ])
            else:
                writer.writerow([
                    "Seq num", "Source", "Queue @src", "Transmit @src",
                    "Receive @C", "Transmit @C", "Receive @D",
                    "End-to-End Delay", "Queuing Delay",
                    "Propagation Delay", "Transmission Delay"
                ])
            writer.writerows(self.packet_records)

    def plot_delay_distribution(self):
            # Extract delays for each source
            delays_A = []
            delays_B = []
            
            for record in self.packet_records:
                if record[1] == 'A':
                    delays_A.append(record[7])  # End-to-End delay
                else:
                    delays_B.append(record[7])

            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(delays_A, bins=20, alpha=0.5, label='Source A', color='blue')
            plt.hist(delays_B, bins=20, alpha=0.5, label='Source B', color='red')
            plt.xlabel('End-to-End Delay (ticks)')
            plt.ylabel('Frequency')
            plt.title('Distribution of End-to-End Delays by Source')
            plt.legend()
            plt.grid(True)
            plt.savefig('delay_distribution.png')
            plt.close()

def single_link_experiment():
    sim = Simulator(max_ticks=1000, transmission_delay=10, propagation_delay=1)
    sim.is_single_link = True  # Set flag for single link experiment
    
    # Create nodes
    A = sim.new_node('A')
    B = sim.new_node('B')
    
    # Set up routes
    A.add_route(B, B)
    
    # Generate two packets at time 0
    packet1 = Packet(A, B)
    packet2 = Packet(A, B)
    
    # Schedule the packets
    sim.schedule_event_after(Event.ENQUEUE, 0, A, packet1)
    sim.schedule_event_after(Event.ENQUEUE, 0, A, packet2)
    
    sim.run()
    sim.write_results('single_link.csv')

def switch_experiment():
    sim = Simulator(max_ticks=10000, transmission_delay=10, propagation_delay=1)
    
    # Create nodes
    A = sim.new_node('A')
    B = sim.new_node('B')
    C = sim.new_switch('C', processing_delay=1)
    D = sim.new_node('D')
    
    # Set up routes
    A.add_route(D, C)
    B.add_route(D, C)
    C.add_route(D, D)
    
    # Schedule packet generation
    # Node A: 10 packets every 1000 ticks
    for burst in range(10):
        for packet in range(10):
            time = burst * 1000
            packet = Packet(A, D)
            sim.schedule_event_after(Event.ENQUEUE, time, A, packet)
    
    # Node B: 2 packets every 500 ticks
    for burst in range(20):
        for packet in range(2):
            time = burst * 500
            packet = Packet(B, D)
            sim.schedule_event_after(Event.ENQUEUE, time, B, packet)
    
    sim.run()
    sim.write_results('switch.csv')
    sim.plot_delay_distribution()
    print("\nDelay Analysis:")
    
    # Calculate average delays
    total_delays = {'A': [], 'B': []}
    queue_delays = {'A': [], 'B': []}
    
    for record in sim.packet_records:
        source = record[1]
        total_delays[source].append(record[7])
        queue_delays[source].append(record[8])
    
    print("\nAverage End-to-End Delays:")
    print(f"Source A: {sum(total_delays['A'])/len(total_delays['A']):.2f} ticks")
    print(f"Source B: {sum(total_delays['B'])/len(total_delays['B']):.2f} ticks")
    
    print("\nAverage Queuing Delays:")
    print(f"Source A: {sum(queue_delays['A'])/len(queue_delays['A']):.2f} ticks")
    print(f"Source B: {sum(queue_delays['B'])/len(queue_delays['B']):.2f} ticks")
    
    print(f"\nPropagation Delay (constant): {sim.propagation_delay * 2} ticks")
    print(f"Transmission Delay (constant): {sim.transmission_delay * 2} ticks")

    # Calculate delay variability
    queue_variability = max(
        max(queue_delays['A']) - min(queue_delays['A']),
        max(queue_delays['B']) - min(queue_delays['B'])
    )
    print("\nDelay Variability Analysis:")
    print(f"Queuing delay varies by up to {queue_variability} ticks")
    print("Queuing delay is the most variable component because:")
    print("1. It depends on network congestion")
    print("2. Varies based on packet arrival timing")
    print("3. Affected by bursts from both sources")

if __name__ == '__main__':
    # Run both experiments
    print("Running single link experiment...")
    single_link_experiment()
    print("Running switch experiment...")
    switch_experiment()
